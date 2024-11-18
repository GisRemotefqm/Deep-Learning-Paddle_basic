"""
使用Softmax函数
假设模型输出10个标签的概率，对应真实标签的概率应该尽可能接近100%，对应其他标签的概率应该接近于0%
且所有概率相加应为1

Softmax(x) = e^xi/(j =0,累加到N)e^xj，i = 0,1,...,C-1
其中C为标签类别个数

交叉熵
    使用的是极大似然思想
    交叉熵的损失函数如下：
    L = -[（k = 1累加到n）tk * log(yk) + (1-tk) * log(1-yk)]
    其中yk代表模型输出，tk代表各个标签，tk中只有正确解的标签为1，其他
"""

# 交叉熵代码
import gzip
import numpy as np
import paddle
import paddle.nn.functional as F
import json
from paddle.nn import Conv2D, MaxPool2D, Linear
from paddle.io import Dataset


class Mnist(paddle.nn.Layer):

    def __init__(self):
        super(Mnist, self).__init__()

        # 定义卷积层、池化层、等
        self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)

        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)

        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)

        # 定义全连接层
        self.fc = Linear(in_features=980, out_features=10)


    def forward(self, inputs):

        output1 = self.conv1(inputs)
        output1 = F.relu(output1)
        output1 = self.max_pool1(output1)

        output2 = self.conv2(output1)
        output2 = F.relu(output2)
        output2 = self.max_pool2(output2)

        output2 = paddle.reshape(output2, [output2.shape[0], 980])
        final_output = self.fc(output2)

        return final_output


# 数据处理
class MnistDataset(Dataset):

    def __init__(self, mode):

        data_file = "./mnist.json.gz"
        data = json.load(gzip.open(data_file))
        train_data, vaild_data, eval_data = data

        if mode == 'train':

            imgs = train_data[0]
            labels = train_data[1]

        elif mode == 'vaild':
            imgs = vaild_data[0]
            labels = vaild_data[1]

        elif mode == 'eval':

            imgs = eval_data[0]
            labels = eval_data[1]

        else:
            raise Exception("please input 'train' or 'valid' or 'eval'")

        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, item):
        img = np.reshape(self.imgs[item], [1, 28, 28]).astype('float32')
        label = np.reshape(self.labels[item], [1]).astype('int64')
        # print("img.shape:", img.shape)
        # print("label.shape", label.shape)

        return img, label

    def __len__(self):
        return len(self.imgs)


# 训练
def train(model):
    print("111111")
    model.train()

    opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    EPOCH_NUM = 10


    for epoch_num in range(EPOCH_NUM):

        for batch_id, data in enumerate(train_loader()):
            img, label = data
            img = paddle.to_tensor(img)
            label = paddle.to_tensor(label)

            # 前向计算
            predict = model(img)

            # 损失函数,使用交叉熵计算损失函数
            loss = F.cross_entropy(predict, label)
            loss_avgs = paddle.mean(loss)

            if batch_id % 200 == 0:
                print("epoch:{}, batch: {}, loss: {} ".format(epoch_num, batch_id, loss_avgs.numpy()))

            # 后向传播
            loss_avgs.backward()
            opt.step()
            opt.clear_grad()

    paddle.save(model.state_dict(), 'cross_entropy_mnist.pdparams')


# 预测
def evaluation(model, datasets):
    model.eval()

    acc_set = list()
    for batch_id, data in datasets:
        imgs, labels = data
        imgs = paddle.to_tensor(imgs)
        labels = paddle.to_tensor(labels)

        predict = model(imgs)

        acc = paddle.metric.accuracy(input=predict, label=labels)
        acc_set.extend((acc.numpy()))

    # 计算多个batch的准去率
    acc_mean = np.array(acc_set).mean()

    return acc_mean


if __name__ == "__main__":

    train_dataset = MnistDataset(mode='train')

    # 实例化一个数据迭代器，异步的
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)

    # vaild_dataset = MnistDataset(model='vaild')

    # vaild_loader = paddle.io.DataLoader(vaild_dataset, batch_size=128, drop_last=True)

    model = Mnist()
    train(model)

