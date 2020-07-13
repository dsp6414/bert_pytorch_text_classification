# -*- coding: UTF-8 -*-
# Created by WangShiYang at 07/11/20

import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """
    配置参数
    """
    def __init__(self, dataset):
        # 模型名称
        self.model_name = "BruceERNIE"
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
        self.bert_path = 'ERNIE_pretrain'
        # bert 切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # 这里的相关的方法，可以通过查看源码来得到
        # bert 隐层个数
        self.hidden_size = 768


class Model(nn.Module):  # 这里就是有些类似于 TF2.0 里边的 tf.keras
    def __init__(self, config):
        """
        构建Bert原生模型
        :param config: 模型的配置参数，模型构建过程中，各个部分高多少，宽多少
        """
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)

        """是否对Bert模型中的参数根据自己的数据进行微调，即对梯度进行调整
            根据自己的需求，看是否需要对Bert的参数进行微调
            通常都是设置为True，进行微调的，用来和自己的业务进行匹配
        """
        for param in self.bert.parameters():
            param.requires_grad = True   # 设置为 True 就是对参数进行微调
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        """
        在代码分析的过程中，要对每一步数据维度的变化进行一个预测，然后和实际运行的结果进行比较
        :param x: [ids, seq_len, mask]
        :return:
        """
        # ids: shape [batch_size, sequence_length]
        # seq_len: 这里传入长度，主要是为了控制每个句子的长度统一
        # mask: 主要是用来挖坑填词理解句意
        context = x[0]  # 对应输入的句子 shape[128, 32]  [batch_size, sequence_length]
        mask = x[2]  # 对应 padding 部分进行 mask shape[128, 32]  [batch_size, sequence_length]

        # pooled：它是在与输入的第一个字符(' CLS ')相关联的隐藏状态之上预先训练的分类器的输出，用于训练下一个句子的任务(参见BERT的论文)。
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)  # shape[128, 768]

        # 原生的 Bert 这里只是简单的接了一个全连接网络分类器
        out = self.fc(pooled)  # shape [128, 10]  输出的是10分类
        return out
