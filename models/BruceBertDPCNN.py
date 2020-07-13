# -*- coding: UTF-8 -*-
# Created by WangShiYang at 07/10/20

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """
    配置参数
    """
    def __init__(self, dataset):
        # 模型名称
        self.model_name = 'BruceBertDPCNN'
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
        # bert 隐藏层个数
        self.hidden_size = 768

        # 上边是 Bert 的参数配置，接下来是 RNN 的参数配置

        # RNN 隐藏层数量
        self.rnn_hidden = 256
        # 卷积核的数量
        self.num_filters = 250

        self.dropout = 0.5


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        # Bert 构建完成之后，接下来构建 DPCNN

        # Conv2D 参数解析：
        # in_channels: 输入的通道数：文本数据，通道数为1
        # out_channels: 输出通道数：250 (论文中的参数设置)
        # kernel_size: 卷积核的大小： 3 (论文中的参数设置)
        # kernel_size是卷积核的大小，有高宽两个部分：这的高设为3(就是类似于N-gram=3，一次选三个词进行卷积)，
        # 宽设置为Bert输出的词向量的维度
        self.conv_region = nn.Conv2d(in_channels=1,
                                     out_channels=config.num_filters,
                                     kernel_size=(3, config.hidden_size))  # 这里使用的是Conv2D
        # 在这里的conv的基础上再接一个conv, 即上一个conv的输出作为这个conv的输入
        self.conv = nn.Conv2d(in_channels=config.num_filters,
                              out_channels=config.num_filters,
                              kernel_size=(3, 1))
        # 到这里为止，就对应着模型图的前两个conv结束了，接下来就是一个block块，里边包含一个池化，两个卷积

        # block

        # 图上最后的一个池化层
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)

        # 为block做准备，定义两个padd   ???
        # (0, 0, 1, 1)表示填充的时候只对后两维(高度和宽度)进行填充，前边的batch_size和channel两个维度不管
        self.padd1 = nn.ZeroPad2d((0, 0, 1, 1))  # 两个维度，所以里边要使用括号
        # (0, 0, 0, 1)表示填充的时候只对最后一维(宽度)进行填充
        self.padd2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()

        # 最后依旧是要接一个线性的分类层
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # 参考BertCNN,这里Bert的输出需要增加一个维度才能输入到卷积层中
        # 在序列模型中 hidden_size = embedding_size
        out = encoder_out.unsqueeze(1)  # [batch_size, 1, seq_len, embed]
        out = self.conv_region(out)  # [batch_size, 250, seq_len-3+1, 1]  seq_len-3+1, 1:是对应的conv里边的计算高宽的公式
        # 对上边的输出结果进行padding,即填充
        out = self.padd1(out)  # [batch_size, 250, seq_len, 1]
        out = self.relu(out)  # Relu不改变维度
        out = self.conv(out)  # [batch_size, 250, seq_len-3+1, 1]  [64, 250, 30, 1]
        out = self.relu(out)  # [64, 250, 30, 1]

        # pytorch里边的 size(),就相当于 tensorflow 里边的 shape()
        while out.size()[2] > 2:  # 即，只要seq_len 维度的值大于2，就一直执行下去
            out = self._block(out)
        # 在这个循环过程中，在最后的时候，基本上后两个维度都是 1
        # 可以看最后 fc 的输入，是两个维度，所以说，最后后边的两个维度的值一定是1，这样才可以去掉
        out = out.squeeze()
        out = self.fc(out)
        return out

    def _block(self, x):
        """
        这里之所以要将 x 传进来，是因为在这里边也要做卷积
        下边注释的维度均为第一次调用block的结果
        函数中的conv的层数也是可以再加的，多加几个也没有什么问题
        :param x: 第一次调用时： [64, 250, 31, 1]  第二次调用时：[64, 250, 15, 1]  7, 3, 1
        :return:
        """
        x = self.padd2(x)  # 1：[64, 250, 31, 1] 2：[64, 250, 16, 1]  ...
        # 这里的block里边第一个是一个pooling(池化)层  是一个Downsampling 下采样，经过下采样之后，得到的 x 应该是维度减小的
        px = self.max_pool(x)  # 1：[64, 250, 15, 1]  2：[64, 250, 7, 1]  ...
        x = self.padd1(px)  # 两位两位的padd 得到[64, 250, 17, 1]  [64, 250, 9, 1]  ...
        x = self.relu(x)
        x = self.conv(x)  # [64, 250, 15, 1]  [64, 250, 7, 1]  ...
        # 每一个conv后边都要接一个padd,以保持维度
        x = self.padd1(x)  # [64, 250, 17, 1]  [64, 250, 9, 1]  ...
        x = self.relu(x)
        x = self.conv(x)  # [64, 250, 15, 1]  [64, 250, 7, 1]  ...
        x = x + px  # [64, 250, 15, 1]  [64, 250, 7, 1]  ...
        return x



