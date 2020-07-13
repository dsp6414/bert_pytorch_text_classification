# -*- coding: UTF-8 -*-
# Created by WangShiYang at 07/07/20

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """
    配置参数
    当参数非常多的时候，通常定义一个config类，在类里边定义参数，传参的时候直接将类传进去即可
    """
    def __init__(self, dataset):
        # 模型名称
        self.model_name = 'BruceBertCNN'
        # 训练集
        self.train_path = dataset + '/data/train.txt'
        # 测试集
        self.test_path = dataset + '/data/test.txt'
        # 校验集
        self.dev_path = dataset + '/data/dev.txt'
        # dataset
        self.datasetpkl = dataset + '/data/dataset.pkl'
        # 类别
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        # 模型训练结果
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 若超过1000个 batch 效果还是没有提升，就提前结束训练
        self.require_improvement = 1000

        # 类别数
        self.num_classes = len(self.class_list)

        # epoch数
        self.num_epochs = 3
        # batch_size
        self.batch_size = 64  # 代码中所有标注的128，因为batch太大了，GPU内存超了，所以全部使用64 代替
        # 每句话处理的长度(短填，长切)
        self.pad_size = 32
        # 学习率
        self.learning_rate = 1e-5
        # bert 预训练模型位置
        self.bert_path = './bert_pretrain'
        # bert 切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # 这里的相关的方法，可以通过查看源码来得到
        # bert 隐层个数
        self.hidden_size = 768

        # 上边是 Bert 的参数配置，接下来是 CNN 的参数配置

        # 卷积核的大小
        self.filter_sizes = (2, 3, 4)  # 类似于N-gram，一次选择几个词聚合在一起
        # 卷积核的数量
        self.num_filters = 256

        self.dropout = 0.5


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 定义Bert的模型结构
        self.bert = BertModel.from_pretrained(config.bert_path)
        # 设置是否对Bert模型原来的参数进行微调
        for param in self.bert.parameters():
            param.requires_grad = True

        # 定义TextCNN的模型结构
        # 构建一个容器，根据filter的大小，存储三个不同尺寸的卷积
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1,   # 原本指输入图片数据的通道数(channel)，这里对于文本数据，没有多通道，维度为1
                       out_channels=config.num_filters,   # 经过卷积层后输出的维度(channel数量)，对应的就是卷积核数量
                       # filter_size = kernel_size
                       # 这里的kernel是2D的(两个维度)，输入的是一个元组类型的数据，这里之所以是两个维度是因为，卷积核是有高和宽的
                       # 这里的高就是一次选几个字(词)进行卷积，而这里的宽就是Bert模型输出的词向量的维度，也就是对应的它的隐藏层的大小
                       # kernel_size的数据样本：(k, Embedding), 这里的 Embedding的维度， 对应的就是模型的 hidden_size 的大小
                       # (2, 768); (3, 768); (4, 768)
                       # 这里对应着从Bert里边出来的数据都是 Embedding 维度(hidden_size)为768
                       # 所以这里卷积核的大小也要设置为768
                       kernel_size=(k, config.hidden_size)) for k in config.filter_sizes]
        )
        self.dropout = nn.Dropout(config.dropout)
        # fc 这个层是都要加的, 需要传入的参数：输入是什么，输出是什么
        # 对于原始的Bert，输入是从Bert中输出的结果，768；输出的就是要分类的数目，num_classes
        # 对于这里的 fc, 输入是卷积的数量乘以len(config.filter_sizes)，前边将的TextCNN，最终输入的就是每对应一个卷积核的大小，就会生成
        # 256个channel，这里总共有三个卷积核的尺 寸，所以得到的是3个256个channel.
        # 输出依旧是，num_classes
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        """
        在这个函数中，将Bert的输出进行相应的卷积和池化
        每调用一次这个函数就是进行一次conv，每次调用的时候输入的 x 是不变的，改变的是创建conv层的kernel_size: 2, 3, 4
        函数总共调用了三次，即进行了三次卷积
        可以看到这里和图像的三次卷积还是不一样的，图像的三次卷积是依次进入三个卷积层，这里是分别进入三个卷积层
        :param x: Bert的输出：out
        :param conv:(1, 256, kernel_size=(2, 768)) (1, 256, kernel_size=(3, 768)) (1, 256, kernel_size=(4, 768))
        :return:
        """
        # x 进入第一个卷积层  x 的维度[64, 1, 32, 768]
        x = conv(x)
        x = F.relu(x)  # Relu 通常不改变形状，conv会改变形状
        # 经过第一个卷积层之后  x 的维度[64, 256, 31, 1]  后边 1 这个维度用不到，将其去掉
        # 这里的最后一个维度，经过卷积之后一定会变成一个1
        x = x.squeeze(3)
        # 通过上边的第一个卷积的过程，将x的最后一个维度去掉了，接下来去掉倒数第二个维度
        # 这里目前的疑惑点就是：为什么要将这几个维度去掉？
        # 这里为什么要把最后一维去掉的解释就是： 因为是一个文本，没有长宽的概念，所以要将最后一维去掉，换成一个三维的
        # 再一个主要的就是，最后输入 fc 网络的时候，要求必须是两维的数据
        # 接下来对这个三维的数据在 31 这个维度上进行 pooling
        size = x.size(2)
        x = F.max_pool1d(x, size)  # 进行 pooling 选择最值保留，其他的值去掉
        x = x.squeeze(2)  # 因为这个时候 31 已经变成 1 了，对于 tensorflow 和 pytorch 来说，当维度为 1 的时候可以省略
        return x

        """上边conv维度变化的内部定义的公式分析：  这里的分析可能存在问题，因为是文本数据
            每次调用函数的计算公式式一样的，只是对应的kernel_size的“input_height”会变化：2, 3, 4
            疑惑：为什么会变成31和1？
            计算公式一：宽的维度31的计算公式：(input_height - kernel_size + 2*padd) / sed[0] + 1
            - 这里的 sed[0] 表示的是横向的移动步长(在文本数据中，通常没有横向的移动步长)
            - 对应公式的计算结果为 (32 - 2 + 2*0) / 1 + 1  这里没有使用 padding，padd 默认为0, 同时由于没有横向移动，所以sed[0] = 1
              = 30 / 1 + 1 = 31
            计算公式二：高的维度1的计算公式：(input_width - kernel_size + 2*padd) / sed[1] + 1
            - 这里的 sed[1] 表示的是竖向的移动步长，这里的纵向的移动步长为 pad_size ????
            - 对应公式的计算结果为 (768 - 768 + 2*0) / 768 + 1 = 0 + 1 = 1 
            对于上边的公式中的 kernel_size,输入的高宽为[32, 768]， 对应的 conv 的 kernel_size 为 [2(3,4), 768]
            所以 input_height 对应的 kernel_size 为2，input_width 对应的 kernel_size 为768 
        """

    def forward(self, x):
        """
        :param x: [ids, seq_len, mask]
        :return:
        """
        context = x[0]  # 对应输入的句子 shape[128, 32]  [batch_size, sequence_length]
        mask = x[2]  # 对应 padding 部分进行 mask shape[128, 32]  [batch_size, sequence_length]

        # pooled: 标记位预测
        # pooled：它是在与输入的第一个字符(' CLS ')相关联的隐藏状态之上预先训练的分类器的输出，用于训练下一个句子的任务(参见BERT的论文)。
        # 这里Debug得到的encoder_out的维度为[batch_size, seq_len, hidden_size],是3维的
        # 但是接下来要输到的卷积层需要的参数确实4维的(最后两维是在一个元组中)，所以接下来要进行维度扩展
        encoder_out, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)  # shape[128, 768]
        # 在batch_size后边扩展一维(输出通道数的维度上进行扩展)  对应的第二个位置上，也就是位置1的维度上
        # 扩展后的维度： [64, 1, 32, 768]  当维度一致的时候，数据就可以输入进去了
        # 对应二维图像来理解：64：样本数量(输入句子个数)；1：通道数；32：高度(每个句子的长度)；768：宽度(每个字(词)的Embedding表示)
        out = encoder_out.unsqueeze(1)

        """接下来探究，上边Bert输出的数据，如何输入到TextCNN中
            首先，由于这里的卷积核的尺寸(filter_size)有三个值，所以这里需要将这三个拼接起来
            Bert输出的维度为：[64, 1, 32, 768]
            卷积的时候不管卷积核的大小是多少，卷积核的数量都是256，对应的前两维都是[64, 256, _, _]
            最后使用的时候，只需要保留前边两维即可，后边的由于是会变化的，所以就不要了，所以拼接的时候也主要看前边的两维
            即三个[128, 256],在第二个位置进行cat的结果为[128, 256*3] = [128, 768]
        """
        # cat(tensors, dim, out),这里的dim，表示的是将前边的tensor在哪一维维度上进行拼接(这里通常在第二维)
        # cat之后输出的维度为：[128, 768]
        # 由于第一个位置是一个tensors,所以要在遍历的结果外边加一个[]
        # 定义一个conv_and_pool函数，将Bert的输出out和conv放到里边，进行相应的卷积和相应的pooling(池化)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  # 经过卷积层后输出的维度为[64, 128]
        out = self.dropout(out)
        # 这里之所以费劲的要变成两维，主要的目的在于：只有变成这样的两维才能输入到 fc 中，才可以进行输出
        out = self.fc(out)  # 在 fc 里边的运算： [64, 768] * [768, 10] = [64, 10]
        return out
