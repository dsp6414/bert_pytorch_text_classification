# -*- coding: UTF-8 -*-
# Created by WangShiYang at 07/08/20

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import utils
from sklearn import metrics
from pytorch_pretrained.optimization import BertAdam


def train(config, model, train_iter, dev_iter, test_iter):
    """
    模型训练方法
    :param config:
    :param model:
    :param train_iter:
    :param dev_iter:
    :param test_iter:
    :return:
    """
    start_time = time.time()  # 获取模型训练开始时间

    # 启动 BatchNormalization 和 Dropout  也就是启动训练模式
    model.train()  # 启动训练模式
    # 拿到 model 的所有参数  debug 可以看到有201个参数
    param_optimizer = list(model.named_parameters())
    # 设置不需要衰减的参数  通常 bias 和 Norm 的参数不需要衰减
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # 给需要衰减的参数定义衰减程度  debug看到需要衰减的参数为77个，不需要的124个
    optimizer_grouped_parameters = [
        # 给需要衰减的参数设置衰减度
        # n: 不需要衰减的参数  p: 需要衰减的参数  最后要的是参数 p  分析的时候从后往前分析
        # any: 是判断元组中有一个不空就为True
        # 如果 nd in n for nd in no_decay 得到的 nd 为空，那么所有不需要衰减的参数都不在里边，整体为True，获取需要衰减的参数
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        # 不需要衰减的参数
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # 定义模型优化器
    optimizer = BertAdam(params=optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)  # 总的迭代次数

    # 定义训练过程
    total_batch = 0  # 记录进行了多少 batch
    dev_best_loss = float('inf')  # 记录校验集最好的 loss  便于我们去评估最好的模型  loss 需要最小
    last_imporve = 0  # 记录上次校验集 loss 下降的 batch 数
    flag = False  # 记录是否很久没有效果提升，停止训练

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}'.format(epoch+1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            # 这里的 i 只起到循环次数的作用, 这里的trains接收的是: (x, seq_len, mask)  labels 接收的数据是: y
            # 将训练数据放到模型中去
            # 将一个批次的数据放入模型中
            outputs = model(trains)
            # 将模型梯度清零  torch 的标准， 手动的清零一下
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()  # 反向传播，更新梯度
            optimizer.step()

            if total_batch % 100 == 0:  # 每多少个 batch 输出在训练集和校验集上的测试效果
                # 这里将测试部分放在 CPU 上进行(也可以在 GPU 上进行)，前边的内容都是放在了 GPU 上，没有进行特别的声明的时候，都是使用 GPU
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                # 这里使用 CPU，主要是因为这里使用的 metrics,是来自 sklearn 的，没有办法使用 GPU 进行计算的，再就是减轻 GPU 的负担
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)  # 在这个方法中对使用校验集数据进行评估
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    # 因为这里更新了最小的 loss，将该 loss 下的模型状态保存一下
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'  # 没有什么实际的作用，* 只是表示已经进行了一次保存
                    last_imporve = total_batch
                else:
                    improve = ''
                time_dif = utils.get_time_dif(start_time)
                # 定义要输出返回的信息  可以用作打印输出的模板
                msg = 'Iter: {0:>6}, Train Loss: {1:>5.2}, Train Acc: {2:>6.2}, Val Loss: {3:>5.2}, ' \
                      'Val Acc:{4:>6.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()  # 上边的操作调用了评估的函数，进入了评估的模式，所以上边代码运行结束后继续进入训练状态
            total_batch = total_batch + 1
            # 这里的训练不是无休止的进行下去的
            if total_batch - last_imporve > config.require_improvement:
                # 这里一定要有模型的终止条件
                # 在验证集上的 loss 超过 1000 个 batch 没有下降，就结束训练
                print('在校验数据集上已经很长时间没有提升了，模型自动停止训练')
                flag = True
                break
        if flag:
            break

    # 这里整个训练过程结束之后就可以在测试集合上进行测试了
    test(config, model, test_iter)


def evaluate(config, model, dev_iter, test=False):  # 标志位默认是False
    """

    :param config:
    :param model:
    :param dev_iter:
    :return:
    """
    model.eval()  # 开启评估模式
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    # 进行评估的时候不需要进行求梯度
    with torch.no_grad():
        # 下边的过程是和训练的过程是一样的
        for texts, labels in dev_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total = loss_total + loss  # 得到全局总的loss

            # 接下来就是要获取总的 acc
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)  # 将 labels append 到 labels_all 里边
            predict_all = np.append(predict_all, predict)

    # 对上边得到的整个的labels和predict，进行整个的计算准确率
    # 上边之所以要转换称为numpy格式，是因为这里要使用CPU进行计算
    acc = metrics.accuracy_score(labels_all, predict_all)

    # 这里acc 后边添加的这些内容主要是为了满足 test 中输出结果的需要
    if test:
        # 这里得到的 report 包含：Precision, Recall and F1-score 三个部分
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(dev_iter), report, confusion  # 这里 test 下的 dev_iter 实际表示的值是test_iter

    return acc, loss_total / len(dev_iter)


def test(config, model, test_iter):
    """
    模型测试
    这个方法其实可以和 evaluate 写成一个方法
    :param config:
    :param model:
    :param test_iter:
    :return:
    """
    # 加载训练得到的模型的最好的状态  即加载训练好的模型
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)  # 设置标志位
    msg = 'Test Loss: {0:>5.2}, Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))

    # 计算 F1 值
    print("Precision, Recall and F1-score")
    print(test_report)
    print('混淆矩阵：\n', test_confusion)
    time_dif = utils.get_time_dif(start_time)
    print('使用时间：', time_dif)



