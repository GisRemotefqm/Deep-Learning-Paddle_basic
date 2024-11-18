import paddle
import os
import random
import paddle.nn.functional as F
from paddle.nn import Conv2D, MaxPool2D, Linear, Dropout
import numpy as np
import cv2 as cv
"""
卷积->激活->池化；卷积->池化->激活
如果使用最大池化，两种顺序是一样的；如果使用平均池化，建议使用第一种顺序
先激活，保留的有效特征值会更多
先平均池化，会有更多的特征值变为0，不利于网络学习
"""
class AlexNet(paddle.nn.Layer):
    def __init__(self,classes):
        super(AlexNet, self).__init__()
        self.conv1 = Conv2D(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=5)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        self.conv3 = Conv2D(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv2D(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = Conv2D(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.max_pool3 = MaxPool2D(kernel_size=2, stride=2)
        self.drop1 = Dropout(p=0.6, mode='downscale_in_infer')
        # 训练时随机丢弃一部分神经元，但是把保留的哪些神经元数值放大，预测时原样输出所有神经元的信息
        self.drop2 = Dropout(p=0.6, mode='upscale_in_train')
        self.fc1 = Linear(in_features=12544, out_features=4096)
        self.fc2 = Linear(in_features=4096, out_features=4096)
        self.fc3 = Linear(in_features=4096, out_features=classes)

    def forward(self, inputs, label=None):

        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.max_pool3(x)

        # 进入需要转成一列数据
        x= paddle.reshape(x, [x.shape[0], -1])


        x = self.fc1(x)
        x = F.relu(x)
        # 随机丢弃一部分神经元，预测时不丢弃
        x = self.drop1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.fc3(x)

        if label is not None:
            acc = paddle.metric.accuracy(x, label)
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
    opt = paddle.optimizer.SGD(learning_rate=0.01, weight_decay=paddle.regularizer.L2Decay(coeff=1e-5), parameters=model.parameters())
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

        paddle.save(model.state_dict(), 'ichallengeAlexNet.pdparams')


# 测试集验证
def evaluation(model):
    use_gpu = True
    paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')

    model_dict = paddle.load('ichallengeAlexNet.pdparams')
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
    model = AlexNet(1)
    train(model)
    evaluation(model)
