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
        self.model_name = 'BruceBertRCNN'
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
        # rnn 数量
        self.num_layers = 2

        self.dropout = 0.5


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        # 在 init 函数中传入的参数都是用来构建模型使用的

        # Bert 构建完成之后，接下来构建 RCNN(这里其实就是使用的双向的LSTM)
        # 参数解析：
        # config.hidden_size： Bert隐藏层的大小
        self.lstm = nn.LSTM(config.hidden_size,
                            config.rnn_hidden,
                            config.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dropout)
        # 这里的池化层中的 kernel_size 参数, 直接使用的是config.pad_size: 每句话的长度，也就是说一次池化就是对一句话的长度进行池化
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]

        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out, _ = self.lstm(encoder_out)
        out = F.relu(out)
        # 上边的LSTM的输出维度为: [64, 32, 512]
        # 按照设置的kernel_size=pad_size=32,来看，这里应该将32移动到最后边   ？？？
        # 对于图像数据来说，形状通常都是[batch, height, width, channels]， 最后一维确实是输出的维度
        # 如果不调换的话，就没有办法将最后一维变成1，按照32的大小对512进行池化，结果是16

        # 对维度进行调换
        out = out.permute(0, 2, 1)
        out = self.maxpool(out)
        out = torch.squeeze(out)  # out.squeeze()
        out = self.fc(out)
        return out
