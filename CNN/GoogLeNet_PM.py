import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D, MaxPool2D, Dropout, Linear, AdaptiveMaxPool2D
import os
import random
import numpy as np
import cv2 as cv

"""

当特征分布较广时，需要较大的卷积核；当特征分布较小时，需要较小的卷积核
所以在选取卷积核时选取合适大小的卷积核是一个问题
GoogLeNet的特点是网络上不仅有深度，在横向上也具有宽度
它提出一种被称为Inception模块的方案，Inception模块采用多通路的设计模式
每个支路使用不同大小的卷积核，最终输出特征图的通道数是每个支路输出通道的总和

"""


class Inception(paddle.nn.Layer):
    """
    其设计思想是使用不同大小的卷积核对图像特征机型提取并附加最大池化操作
    最后将特征沿着通道这一维度进行拼接，最后输出的特征图包含不同大小的卷积
    核提取到的特征，简单的多通路拼接会使通道数迅速增长，使用1 * 1的卷积控制通道数

    """
    def __init__(self, c0, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__()
        self.p1_1 = Conv2D(in_channels=c0, out_channels=c1, kernel_size=1)
        self.p2_1 = Conv2D(in_channels=c0, out_channels=c2[0], kernel_size=1)
        self.p2_2 = Conv2D(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)
        self.p3_1 = Conv2D(in_channels=c0, out_channels=c3[0], kernel_size=1)
        self.p3_2 = Conv2D(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)
        self.p4_1 = MaxPool2D(kernel_size=3, stride=1, padding=1)
        self.p4_2 = Conv2D(in_channels=c0, out_channels=c4, kernel_size=1)

    def forward(self, inputs):

        # 支路1包含一个1*1的卷积
        p1 = F.relu(self.p1_1(inputs))

        # 支路2包含1*1的卷积和3*3的卷积
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(inputs))))

        # 支路三包含1*1的卷积和5*5的卷积
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(inputs))))

        # 支路4包含最大池化和1*1的卷积
        p4 = F.relu(self.p4_2(F.relu(self.p4_1(inputs))))

        return paddle.concat([p1, p2, p3, p4], axis=1)


class GoogLeNet(paddle.nn.Layer):

    def __init__(self, classes):
        super(GoogLeNet, self).__init__()

        self.conv1 = Conv2D(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.max_pool1 = MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 第二个模块
        self.conv2_1 = Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.conv2_2 = Conv2D(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1)
        self.max_pool2 = MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 第三个模块
        self.block3_1 = Inception(192, 64, (96, 128), (16, 32), 32)
        self.block3_2 = Inception(256, 128, (128, 192), (32, 96), 64)
        self.max_pool3 = MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 第四个模块
        self.block4_1 = Inception(480, 192, (96, 208), (16, 48), 64)
        self.block4_2 = Inception(512, 160, (112, 224), (24, 64), 64)
        self.block4_3 = Inception(512, 128, (128, 256), (24, 64), 64)
        self.block4_4 = Inception(512, 112, (144, 288), (32, 64), 64)
        self.block4_5 = Inception(528, 256, (160, 320), (32, 128), 128)
        self.max_pool4 = MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 第五个模块
        self.block5_1 = Inception(832, 256, (160, 320), (32, 128), 128)
        self.block5_2 = Inception(832, 384, (192, 384), (48, 128), 128)
        self.max_pool5 = AdaptiveMaxPool2D(output_size=1)
        self.fc = Linear(in_features=1024, out_features=classes)

    def forward(self, inputs, label=None):

        x = self.max_pool1(F.relu(self.conv1(inputs)))
        x = self.max_pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(x)))))
        x = self.max_pool3(self.block3_2(self.block3_1(x)))
        x = self.block4_3(self.block4_2(self.block4_1(x)))
        x = self.max_pool4(self.block4_5(self.block4_4(x)))
        x = self.max_pool5(self.block5_2(self.block5_1(x)))
        x = paddle.reshape(x, [x.shape[0], -1])

        x = self.fc(x)

        if label is not None:
            acc = paddle.metric.accuracy(x, label= label)
            return x, acc

        else:
            return x


# 对数据进行预处理
def transform_img(img):

    # 将图片缩放为224*224
    img = cv.resize(img, (224, 224))

    # 输入的图像为（H, W, C）,需要转为[C, H, W]
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32')

    # 对数据进行归一化处理, 并将范围调整到[-1, 1]之间
    img = img /255.0
    img = img * 2 - 1

    return img


# 数据读取器
def data_loader(file_dir, batch_size=10, mode='train'):

    # 读取该路径下所有的文件名
    filenames = os.listdir(file_dir)

    def reader():

        if mode == 'train':
            random.shuffle(filenames)

        img_list = []
        label_list = []

        for name in filenames:

            # 生成文件路径
            filepath = os.path.join(file_dir, name)
            img = cv.imread(filepath)
            img = transform_img(img)

            # H开头表示高度近视，N开头的文件表示正常视力
            if name[0] == "H" or name[0] == 'N':
                label = 0
            # P开头的文件表示病理性近视
            elif name[0] == 'P':
                label = 1
            else:
                raise Exception('错了')

            img_list.append(img)
            label_list.append(label)

            if len(img_list) == batch_size:
                yield np.array(img_list).astype('float32'), np.array(label_list).astype('float32').reshape(-1, 1)
                img_list = []
                label_list = []

        if len(img_list) > 0:

            yield np.array(img_list).astype('float32'), np.array(label_list).astype('float32').reshape(-1, 1)

    return reader


# 验证集数据读取
def vaild_data_loader(filedir, csvfile, batch_size=10, mode='vaild'):

    lines = open(csvfile).readlines()
    def reader():
        img_list = []
        label_list = []
        for line in lines[1:]:
            line = line.strip().split(',')
            if line[1] == "" or line[1] == "":
                continue
            name = line[1]

            label = int(line[2])
            filepath = os.path.join(filedir, name)
            img = cv.imread(filepath)
            img = transform_img(img)

            img_list.append(img)
            label_list.append(label)

            if len(img_list) == batch_size:
                yield np.array(img_list).astype('float32'), np.array(label_list).astype('float32').reshape(-1, 1)
                img_list = []
                label_list = []

        if len(img_list) > 0:
            yield np.array(img_list).astype('float32'), np.array(label_list).astype('float32').reshape(-1, 1)

    return reader


def train(model):
    model.train()
    opt = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9, parameters=model.parameters())
    EPOCH_NUM = 10
    train_loader = data_loader(train_dir, batch_size=10, mode='train')
    vaild_loader = vaild_data_loader(vaild_dir, csvfile, batch_size=10, mode='vaild')

    for epoch_id in range(EPOCH_NUM):
        for iter_id, data in enumerate(train_loader()):
            imgs, labels = data
            imgs = paddle.to_tensor(imgs)
            labels = paddle.to_tensor(labels)

            predict = model(imgs)

            loss = F.binary_cross_entropy_with_logits(predict, labels)
            avgs_loss = paddle.mean(loss)

            if iter_id % 20 == 0:
                print("epoch_id: {}, batch_id: {}, loss is: {:.4f}".format(epoch_id, iter_id, float(avgs_loss.numpy())))

            avgs_loss.backward()
            opt.step()
            opt.clear_grad()

        model.eval()
        acc = []
        losses = []
        for batch_id, data in enumerate(vaild_loader()):
            x, y = data
            x = paddle.to_tensor(x)
            y = paddle.to_tensor(y)

            logits = model(x)
            # 二分类，sigmoid计算结果以0.5为阈值
            pred = F.sigmoid(logits)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            # 计算预测概率小于0.5的类别
            pred2 = pred * (-1.0) + 1.0
            # 得到两个类别的预测概率，并沿着第一个维度级联
            pred = paddle.concat([pred2, pred], axis=1)
            accuracy = paddle.metric.accuracy(pred, paddle.cast(y, dtype='int64'))
            acc.append(accuracy.numpy())
            losses.append(loss.numpy())
        print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(acc), np.mean(losses)))
        model.train()

        paddle.save(model.state_dict(), 'ichallengeLeNet.pdparams')


# 测试集验证
def evaluation(model):
    use_gpu = True
    paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')

    model_dict = paddle.load('ichallengeLeNet.pdparams')
    model.load_dict(model_dict)

    model.eval()
    eval_loader = data_loader(eval_dir, batch_size=10, mode='eval')

    acc_set = []
    avgs_loss_set = []
    for batch_id, data in enumerate(eval_loader()):
        x, y = data
        x = paddle.to_tensor(x)
        label = paddle.to_tensor(y)
        y = y.astype(np.int64)
        y_tensor_int64 = paddle.to_tensor(y)

        predict, acc = model(x, y_tensor_int64)
        loss = F.binary_cross_entropy_with_logits(predict, label)
        avgs_loss = paddle.mean(loss)

        avgs_loss_set.append(avgs_loss.numpy())
        acc_set.append(acc.numpy())

    acc_mean = np.array(acc_set).mean()
    avgs_loss_mean = np.array(avgs_loss_set).mean()

    print('loss = {:.4f}, acc = {: .4f}'.format(avgs_loss_mean, acc_mean))


if __name__ == "__main__":
    train_dir = './training/PALM-Training400/PALM-Training400/PALM-Training400'
    vaild_dir = './validation/PALM-Validation400'
    csvfile = './valid_gt/PALM-Validation-GT/PM_Label_and_Fovea_Location.csv'
    eval_dir = './training/PALM-Training400/PALM-Training400/PALM-Training400'
    model = GoogLeNet(1)
    train(model)
    evaluation(model)
