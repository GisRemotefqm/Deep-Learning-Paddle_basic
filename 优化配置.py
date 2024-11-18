"""
PRC通信方式，常常用于CPU分布式训练，有两个节点，一个是Paramet Service和训练节点Trainer

NCCL2通信方式, 不需要启动Parameters Service
"""

import paddle
import numpy as np
import random
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F
import gzip
import json


def load_data(mode= "train"):

    file_path = "./mnist.json.gz"
    data = json.load(gzip.open(file_path))
    train_data, vaild_data, eval_data = data

    if mode == 'train':
        imgs, labels = train_data[0], train_data[1]

        return np.array(imgs), np.array(labels)

    elif mode == 'vaild':
        imgs, labels = vaild_data[0], vaild_data[1]
        return np.array(imgs), np.array(labels)

    elif mode == 'eval':

        imgs, labels = eval_data[0], eval_data[1]
        return np.array(imgs), np.array(labels)
    else:
        raise Exception('should input "train" or "vaild" or "eval"')


def data_loader():

    imgs, labels = load_data('train')

    index_list = list(range(len(imgs)))
    random.shuffle(index_list)
    img_list = []
    label_list = []
    EPOCH_NUM = 100

    for i in index_list:
        img = np.reshape(imgs[i], [1, 28, 28]).astype('float32')
        label = np.reshape(labels[i], [1]).astype('int64')

        img_list.append(img)
        label_list.append(label)

        if len(img_list) == EPOCH_NUM:

            yield np.array(img_list), np.array(label_list)
            img_list = []
            label_list = []

        if len(img_list) > 0:
            yield np.array(img_list), np.array(label_list)

    return data_loader


class Mnist(paddle.nn.Layer):

    def __init__(self):
        super(Mnist, self).__init__()
        self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)

        self.fc = Linear(in_features=980, out_features=10)


    def forward(self, inputs):

        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x, [x.shape[0], 980])
        x = self.fc(x)

        return x

def train(model):

    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

    model.train()
    loader = data_loader
    # 四种优化算法方案
    # 随机梯度下降
    opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    # 引入物理动量概念，累计速度，减少动荡，使参数更新方向稳定
    opt = paddle.optimizer.Momentum(learning_rate=0.01, parameters=model.parameters())
    # 学习率逐渐下降，根据参数变化大小调整学习率
    opt = paddle.optimizer.Adagrad(learning_rate=0.01, parameters=model.parameters())
    # 将动量和自适应学习两个优化思路结合起来
    opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())

    EPOCH_NUM = 5

    for epoch_id in range(EPOCH_NUM):
        for iter_id, data in enumerate(loader()):

            img, label = data

            img = paddle.to_tensor(img)
            label = paddle.to_tensor(label)

            # 前向计算
            predict = model(img)

            # 计算误差
            loss = F.cross_entropy(predict, label)
            avgs_loss =paddle.mean(loss)

            if iter_id % 200 == 0:
                print("epoch_id: {}, iter_id: {}, loss: {}".format(epoch_id, iter_id, avgs_loss.numpy()))

            # 后向传播
            avgs_loss.backward()
            opt.step()
            opt.clear_grad()


if __name__ == "__main__":
    mnist = Mnist()

    train(mnist)

