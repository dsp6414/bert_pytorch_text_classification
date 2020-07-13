# -*- coding: UTF-8 -*-
# Created by WangShiYang at 07/10/20

import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """
    配置参数
    """
    def __init__(self, dataset):
        # 模型名称
        self.model_name = 'BruceBertRNN'
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
        # 这里就是相当于Bert这个模型类的实例化的过程
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        # Bert 构建完成之后，接下来构建 RNN(通常使用LSTM或者GRU, 这里使用的是双向的LSTM)

        # 参数分析
        # input_size: 放入的就是Bert模型输出的内容  congfig.hidden_size = 768
        # hidden_size: RNN(LSTM) 本身的隐藏层的数量   congfig.rnn_hidden = 256
        # num_layers: LSTM 的层数
        # batch_first: 按照指定的方式输入和输出内容  (batch, seq, feature(hidden_size))这样的一个tensor
        # bidirectional: 设置是否使用双向的LSTM
        self.lstm = nn.LSTM(config.hidden_size,
                            config.rnn_hidden,
                            config.num_layers,
                            batch_first=True,
                            dropout=config.dropout,
                            bidirectional=True)

        self.dropout = nn.Dropout(config.dropout)

        # 因为是双向的LSTM，所以参数翻倍
        self.fc = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def forward(self, x):
        """
        :param x: [ids, seq_len, mask]
        :return:
        """
        context = x[0]  # 对应输入的句子 shape[128, 32]  [batch_size, sequence_length]
        mask = x[2]  # 对应 padding 部分进行 mask shape[128, 32]  [batch_size, sequence_length]

        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)

        # 分析源码可以看出：LSTM模型的返回值是两个参数： output, (h_n, c_n)，这里只使用output就可以了(用啥就留啥)
        # out: (batch, seq_len, num_directions * hidden_size)
        out, _ = self.lstm(encoder_out)
        out = self.dropout(out)
        # 这里要传入fc的out的维度为：[64, 32, 512], 不符合要求，要求的维度为：[64, 512]
        # 使用squeeze去掉维度要求只能去掉维度为 1 的维度，现在是32，去不掉
        # 这里使用切片的方法：第一个维度和第三个维度的都要，中间第二个维度不要，很多模型中去掉某一个维度，很多时候就是为了ping维度
        out = out[:, -1, :]  # -1 就表示不要了

        out = self.fc(out)
        return out


