import paddle
import numpy as np
import paddle.nn.functional as F
from paddle.nn import Conv2D, MaxPool2D, Dropout, Linear
import cv2 as cv
import os
import random


# ResNet使用来BatchNorm层，在卷积层后加入Batchnorm用来提高数值稳定性
class ConvBNLayer(paddle.nn.Layer):
    def __init__(self, channels, out_filter, k_size, stride=1, group=1, act=None):
        super(ConvBNLayer, self).__init__()
        self.conv = Conv2D(
            in_channels=channels,
            out_channels=out_filter,
            kernel_size=k_size,
            stride=stride,
            padding=(k_size - 1) // 2,
            groups=group,
            bias_attr=False)

        # BatchNorm提升数值稳定性，加在卷积层后面
        self.batch_norm = paddle.nn.BatchNorm2D(out_filter)
        self.act = act

    def forward(self, inputs):

        y = self.conv(inputs)
        y = self.batch_norm(y)

        if self.act == 'leaky':
            y = F.leaky_relu(x=y, negative_slope=0.1)
        elif self.act == 'relu':
            y = F.relu(x=y)

        return y


# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
class BottleneckBlock(paddle.nn.Layer):

    def __init__(self,
                 in_ch,
                 out_filter,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()

        self.conv1 = ConvBNLayer(
            channels=in_ch,
            out_filter=out_filter,
            k_size=1,
            act='relu')

        self.conv2 = ConvBNLayer(
            channels=out_filter,
            out_filter=out_filter,
            k_size=3,
            stride=stride,
            act='relu')

        self.conv3 = ConvBNLayer(
            channels=out_filter,
            out_filter=out_filter * 4,
            k_size=1,
            act=None)

        # 如果第三个卷积层的输出和此残差块的输入数据一致，则shortcut=True
        # 否则shortcut= False，添加一个1 * 1的卷积，使其形状与第三个卷积层一直
        if not shortcut:
            self.short = ConvBNLayer(
                channels=in_ch,
                out_filter=out_filter * 4,
                k_size=1,
                stride=stride)

        self.shortcut = shortcut
        self._num_channels_out = out_filter * 4

    def forward(self, inputs):

        y = self.conv1(inputs)
        conv2 = self.conv2(y)
        conv3 = self.conv3(conv2)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv3)
        y = F.relu(y)

        return y


class ResNet(paddle.nn.Layer):

    def __init__(self, layers=50, class_dim=1):
        super(ResNet, self).__init__()
        self.layers = layers
        layers_list = [50, 101, 152]
        assert layers in layers_list, 'supported layers are {} but input '.format(layers_list, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]

        # 残差中使用到的输出通道数
        out_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(channels=3, out_filter=64, k_size=7, stride=2, act='relu')
        self.max_pool = MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 第二到第五个模块c2, c3, c4, c5
        self.bottleneck_block_list = []
        num_channels = 64

        for block in range(len(depth)):

            shortcut = False
            for i in range(depth[block]):
                print('bb_%d_%d' % (block, i))
                # c3, c4, c5会在第一个残差块使用stride=2，其余所有残差块stride=1
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        in_ch=num_channels,
                        out_filter=out_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # 在c5的输出特征图中使用全局池化
        self.poo2d_avg = paddle.nn.AdaptiveAvgPool2D(output_size=1)

        # stdv作为全连接层随机初始化的方差
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        # 创建全连接层，输出大小为类别数目，经过残差网络的卷积和全局池化后
        # 卷积特省的维度事[B, 2048, 1, 1]，故最后一层全连接的输入维度事2048
        self.out = Linear(in_features=2048, out_features=class_dim, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):

        y = self.conv(inputs)
        y = self.max_pool(y)

        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)

        y = self.poo2d_avg(y)
        y = paddle.reshape(y, [y.shape[0], -1])
        y = self.out(y)

        return y


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
    opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters(),
                                    weight_decay=0.001)
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

        paddle.save(model.state_dict(), 'ichallengeVGG.pdparams')


# 测试集验证
def evaluation(model):
    use_gpu = True
    paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')

    model_dict = paddle.load('ichallengeVGG.pdparams')
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
    model = ResNet()
    train(model)
    evaluation(model)

