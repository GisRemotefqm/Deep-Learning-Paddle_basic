import paddle
import paddle.nn.functional as F
import numpy as np
from paddle.nn import Conv2D, MaxPool2D, Linear
import random
import os
import cv2 as cv


class LeNet1(paddle.nn.Layer):

    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = Conv2D(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)

        self.conv3 = Conv2D(in_channels=16, out_channels=120, kernel_size=4, stride=1)
        self.fc1 = Linear(in_features=300000, out_features=64)
        self.fc2 = Linear(in_features=64, out_features=1)

    def forward(self, inputs, label=None):
        x = self.conv1(inputs)
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = F.sigmoid(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)

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
    train_loader = data_loader(train_dir, 10, 'train')
    vaild_loader = vaild_data_loader(vaild_dir, csvfile, batch_size=10)

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
    model = LeNet1()
    train(model)
    evaluation(model)

