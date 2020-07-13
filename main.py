# -*- coding: UTF-8 -*-
# Created by WangShiYang at 07/07/20

import time
import torch
import argparse
import numpy as np
import utils
import train
# 下边是一种动态导入的形式，可以根据模型进行动态加载，将很多的模型写在一块的时候，通常使用这个模块
from importlib import import_module


parser = argparse.ArgumentParser(description='Bruce-Bert-Text-Classification')
parser.add_argument('--model', type=str, default='BruceBert', help='choose a model')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集地址
    model_name = args.model
    x = import_module('models.' + model_name)
    # debug 的过程中，进入到模型中的时候不用点击进入，在模型中打 好断点直接，下一步即可进入到模型中，可能不会自己跳转进去，但是可以自己点进去看看
    config = x.Config(dataset)  # debug 到这里的时候，自动走到模型的 __init__ 里边了

    # 为保证每次运行的结果一样，进行一些设置
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print('加载数据集')
    train_data, dev_data, test_data = utils.build_dataset(config)

    # 加载完数据之后，构造迭代器，将数据一个批次一个批次的送入神经网络中
    train_iter = utils.build_iterator(train_data, config)
    dev_iter = utils.build_iterator(dev_data, config)
    test_iter = utils.build_iterator(test_data, config)

    time_dif = utils.get_time_dif(start_time)
    print('模型开始前准备数据的时间：', time_dif)

    # 这里上边准备完了数据之后，就可以直接调用模型进行训练, 评估和测试了
    model = x.Model(config).to(config.device)
    # 训练 + 验证 + 测试  三者都会运行
    train.train(config, model, train_iter, dev_iter, test_iter)
    # 测试  只运行测试(设计验证部分)
    # train.test(config, model, test_iter)

