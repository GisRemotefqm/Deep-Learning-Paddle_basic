import paddle
import paddle.nn.functional as F
import numpy as np
from paddle.nn import Conv2D, MaxPool2D, Linear
import json
import gzip
import random


class LeNet1(paddle.nn.Layer):

    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)

        self.conv3 = Conv2D(in_channels=16, out_channels=120, kernel_size=4, stride=1)
        self.fc1 = Linear(in_features=120, out_features=64)
        self.fc2 = Linear(in_features=64, out_features=10)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)

        return x


def load_data(mode):
    file_path = '../mnist.json.gz'
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


def train_loader(mode):
    imgs, labels = load_data(mode)
    list_index = list(range(len(imgs)))
    if mode == 'train':
        random.shuffle(list_index)

    EPOCH_NUM = 100
    img_list = []
    label_list = []

    for i in list_index:
        img = np.array(imgs[i]).astype('float32')
        label = np.array(labels[i]).astype('int64')

        img = img.reshape([1, 28, 28]).astype('float32')
        label = label.reshape([1]).astype('int64')

        img_list.append(img)
        label_list.append(label)

        if len(img_list) == EPOCH_NUM:
            yield np.array(img_list), np.array(label_list)
            img_list = []
            label_list = []

    if len(img_list) > 0:
        yield np.array(img_list), np.array(label_list)

    return train_loader


def train(model):


    model.train()
    opt = paddle.optimizer.SGD(learning_rate=0.1, weight_decay=paddle.regularizer.L2Decay(coeff=1e-5),
                               parameters=model.parameters())
    EPOCH = 10

    for epoch_id in range(EPOCH):
        for iter_id, data in enumerate(train_loader(mode)):

            imgs, labels = data
            imgs = paddle.to_tensor(imgs)
            labels = paddle.to_tensor(labels)
            # 前向计算
            predict = model(imgs)

            loss = F.cross_entropy(predict, labels)
            avgs_loss = paddle.mean(loss)

            if iter_id % 200 == 0:
                print('epoch_id: {}, iter_id: {}, loss: {}'.format(epoch_id, iter_id, avgs_loss.mean()))

            avgs_loss.backward()
            opt.step()
            opt.clear_grad()

    paddle.save(model.state_dict(), "cnnmnist.pdparams")


def evlation(model):

    print('start evaluation...')
    params_path = 'cnnmnist.pdparams'
    params_dict = paddle.load(params_path)
    model.load_dict(params_dict)

    model.eval()

    for batch_id, data in enumerate(train_loader(mode)):
        imgs, labels = data

        imgs = paddle.to_tensor(imgs)
        labels = paddle.to_tensor(labels)

        predict = model(imgs)

        loss = F.cross_entropy(predict, labels)
        avgs_loss = paddle.mean(loss)

        # print('predict is ', predict)
        print('labels is ', labels)
        print('avgs_loss is ', avgs_loss.numpy())





if __name__ == "__main__":
    model = LeNet1()
    # mode = 'train'
    # train(model)
    mode = 'eval'
    evlation(model)
