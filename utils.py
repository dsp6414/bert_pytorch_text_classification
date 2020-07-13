# -*- coding: UTF-8 -*-
# Created by WangShiYang at 07/07/20

from tqdm import tqdm
import torch
import time
import pickle as pkl
import os
from datetime import timedelta


PAD, CLS = '[PAD]', '[CLS]'


def load_dataset(file_path, config):
    """
    定义一个函数的时候，要想清楚，最后需要的返回结果是什么，定义结束后，通过debug来检查是否是得到了想要的结果
    返回的结果就是要将数据处理成四个list:  ids, label, ids_len, mask
    :param file_path:
    :param seq_len:
    :return:
    """
    contents = []
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):  # 用于显示数据处理进度
            line = line.strip()  # 去掉句子两端的空格
            if not line:
                continue
            # 分离文本内容和标签
            content, label = line.split('\t')  # 原始数据是使用制表符 '\t' 分隔文本内容和 label 的
            # 对文本内容进行处理
            token = config.tokenizer.tokenize(content)  # 使用 Bert 中的 tokenizer 将句子切分开来
            # 根据 Bert 输入数据的格式，添加相关的标志位
            token = [CLS] + token
            seq_len = len(token)
            mask = []  # 保证长度一致
            token_ids = config.tokenizer.convert_tokens_to_ids(token)  # 将词转换为对应的 id

            pad_size = config.pad_size
            if pad_size:
                if len(token) < pad_size:
                    # 对 mask 进行填充，前边有数据的部分填充 1 ，后边没有数据的部分填充 0
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    # 短填, 这里短的部分填充0
                    token_ids = token_ids + ([0] * (pad_size - len(token)))
                else:  # 长切
                    mask = [1] * pad_size  # 每个位置都有数据
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, int(label), seq_len, mask))
    return contents  # 返回的是train, dev, test三个部分对应的输入到模型的数据，后边再进行一些处理，划分一下批次就可以送入模型了


def build_dataset(config):
    """
    返回的三个部分的数据，每个部分都对应着四个list: ids, label, ids_len, mask
    由于每次调试的时候都需要加载数据，十分费劲，这里进行一个改进，这里只需要将加载好的数据保存成一个pkl文件即可
    :param config:
    :return: train, dev, test
    """
    # 首先判断是否存在pkl文件
    if os.path.exists(config.datasetpkl):
        # 第一次加载数据之后，以后每次加载数据就运行这里的代码
        dataset = pkl.load(open(config.datasetpkl, 'rb'))  # 打开的是二进制文件，所以使用 'rb' 的方式
        # 由于存储的时候存储的是一个字典，所以读取的时候依旧需要按照字典的形式读取
        train = dataset['train']
        dev = dataset['dev']
        test = dataset['test']
    else:
        # 第一次加载数据的时候运行这一部分代码，并对加载好的数据进行保存
        train = load_dataset(config.train_path, config)
        dev = load_dataset(config.dev_path, config)
        test = load_dataset(config.test_path, config)
        # 这里将三个值同时保存起来，使用字典 dict 设置键值对
        dataset = {}
        dataset['train'] = train
        dataset['dev'] = dev
        dataset['test'] = test
        # 将导入的数据保存成字典格式后，以二进制的形式写入到.pkl文件中
        pkl.dump(dataset, open(config.datasetpkl, 'wb'))
    return train, dev, test


class DatasetIterator(object):
    def __init__(self, dataset, batch_size, device):
        self.datatset = dataset
        self.batch_size = batch_size
        self.n_batches = len(dataset) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(dataset) % self.n_batches != 0:
            self.residue = True
        self.index = 0  # 传入数据的批次标记
        self.device = device

    def _to_tensor(self, datas):
        """
        拿到预处理后得到的数据，选择其中一部分，并将其转换为Tensor数据，用来输入到Bert中
        在使用GPU的时候，传入的数据再送到模型之前要将其转换为tensor，否则运行不了
        :param datas: 这里传入的数据包含4个list: ids, label, ids_len, mask
        :return: 要输入到Bert的数据单独包在一个元组里边: (x, seq_len, mask), 这里返回的标签不传入到模型: y
        """
        # 这里的 x 的维度是 [128, 32] 所以下边加一个[]，作用是使之变成一个二维的数据，下同
        # 不加的话只是一个一维的数据：128个list
        x = torch.LongTensor([item[0] for item in datas]).to(self.device)  # 获取第一个list  x: 样本数据ids
        y = torch.LongTensor([item[1] for item in datas]).to(self.device)  # 标签数据label

        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device)  # 每一个序列的真实长度
        mask = torch.LongTensor([item[3] for item in datas]).to(self.device)

        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:  # 数据进入最后一个批次时：
            batches = self.datatset[self.index * self.batch_size: len(self.datatset)]  # 最后一个批次的数据
            self.index += 1  # 这里加1之后self.index 就不等于 self.n_batches 了，跳出
            # 这里得到的是上边数据预处理方法返回的原生的数据，里边包含四个list，这里接下来还要处理成Bert需要的形式：[ids, seq_len, mask]
            batches = self._to_tensor(batches)  # 因为这里分完批次之后数据就要直接送到模型中了，所以要将数据转换为Bert模型需要的类型
            return batches
        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.datatset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterator(dataset, config.batch_size, config.device)  # 在上边自己构建数据集的迭代器类
    return iter


def get_time_dif(start_time):
    """
    获取已经使用的时间
    :param start_time:
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

