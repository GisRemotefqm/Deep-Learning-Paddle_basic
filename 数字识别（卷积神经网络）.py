"""
卷积神经网络包括多个卷积层和池化层组成

"""
import paddle
from paddle.nn import Linear, Conv2D, MaxPool2D
import random
import os
import json
import gzip
import paddle.nn.functional as F
import numpy as np

def load_data(mode = 'train'):

    file_path = './mnist.json.gz'
    data = json.load(gzip.open(file_path))
    train_data, valid_data, evla_data = data

    if mode == 'train':
        img = train_data[0]
        label = train_data[1]

        return img, label
    elif mode == 'eval':
        img = evla_data[0]
        label = evla_data[1]

        return img, label
    elif mode == 'valid':
        img = valid_data[0]
        label = valid_data[1]

        return img, label
    else:
        raise Exception("mode = 'train', 'eval', 'valid")


class MNIST(paddle.nn.Layer):

    def __init__(self):
        super(MNIST, self).__init__()

        # 定义卷积层
        # 输出特征通道为20，卷积核大小为5，卷积步长为1，padding=2
        self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层,池化核大小为2，池化步长为2
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 定义卷积层
        self.conv2 = Conv2D(in_channels=20,out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)

        # 定义全连接层
        self.fc = Linear(in_features=980, out_features=128)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x


# 数据生成器
def data_generator():

    imgs, labels = load_data()

    # 通过索引打乱顺序
    length = len(imgs)
    index_list = list(range(length))
    random.shuffle(index_list)

    # 获得数据
    BATCH_SIZE = 25
    img_list = []
    label_list = []
    img_r = 28
    img_h = 28

    for i in index_list:


        img = np.array(imgs[i]).astype('float32')
        label = np.array(labels[i]).astype('float32')

        # 需要图像信息所以需要将其改为28 * 28
        img = img.reshape([1, img_r, img_h]).astype('float32')
        # print(img.shape)
        label = label.reshape([1]).astype('float32')

        img_list.append(img)
        label_list.append(label)

        if len(img_list) == BATCH_SIZE:

            yield np.array(img_list), np.array(label_list)
            img_list = []
            label_list = []

    if len(img_list) > 0:
        yield np.array(img_list), np.array(label_list)

    return data_generator

def train(model):

    model.train()
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())

    EPOCH_NUM = 40
    for epoch_id in range(EPOCH_NUM):

        generator = data_generator
        for iter_id, data in enumerate(generator()):

            # 转为动态图
            img_list, label_list = data
            print("label_list", label_list.shape)
            tensor_img = paddle.to_tensor(img_list)
            tensor_label = paddle.to_tensor(label_list)

            # 向前计算
            predict = model(tensor_img)

            # 计算损失
            loss = F.square_error_cost(predict, tensor_label)
            avgs_loss = paddle.mean(loss)

            if iter_id % 200 == 0:

                print("epoch:{}, batch:{}, loss:{}".format(epoch_id, iter_id, avgs_loss))

            # 向后计算
            avgs_loss.backward()
            opt.step()
            opt.clear_grad()

    # 保存模型
    paddle.save(model.state_dict(), './mnist2.pdparams')

if __name__ == "__main__":

    model = MNIST()

    train(model)