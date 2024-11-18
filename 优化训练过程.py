"""
优化训练过程有五个环节
1. 计算分类准确率，观测模型训练效果
2. 检查模型训练过程
3，加入效验或测试，根号请假模型效果
4. 加入正则化项，避免过拟合
5. 可视化分析

"""
import matplotlib.pyplot as plt

"""
过拟合原因
    1.是由于模型过于敏感，训练数据量太少或其中噪音太多
    2.使用强大模型的同时训练数据太少，导致在训练数据上表现好的候选假设太多

正则化
    在没有扩充样本量的可能下，只能降低模型复杂度，可以通过限制参数的数量或者可能取值实现

"""

# 加载相关库
import random
import paddle
import numpy as np
import gzip
import json
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F


# 定义数据集读取器
def load_data(mode='train'):
    # 读取数据文件
    datafile = './mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    # 读取数据集中的训练集，验证集和测试集
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28
    # 根据输入mode参数决定使用训练集，验证集还是测试
    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]
    # 获得所有图像的数量
    imgs_length = len(imgs)
    # 验证图像数量和标签数量是否一致
    assert len(imgs) == len(labels), \
        "length of train_imgs({}) should be the same as train_labels({})".format(
            len(imgs), len(labels))

    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        # 训练模式下，打乱训练数据
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        # 按照索引读取数据
        for i in index_list:
            # 读取图像和标签，转换其尺寸和类型
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img)
            labels_list.append(label)
            # 如果当前数据缓存达到了batch size，就返回一个批次数据
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据缓存列表
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator


use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
# 用作可视化分析，将训练批次编号作为x轴坐标，对应批次训练损失作为纵坐标



# 仅优化算法的设置有所差别
def train(model):
    num = 0
    iters = []
    losses = []
    model = MNIST()
    model.train()

    # 四种优化算法的设置方案，可以逐一尝试效果,weight_decay加入正则化项
    opt = paddle.optimizer.SGD(learning_rate=0.01, weight_decay=paddle.regularizer.L2Decay(coeff=1e-5),parameters = model.parameters())
    # opt = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9, parameters=model.parameters())
    # opt = paddle.optimizer.Adagrad(learning_rate=0.01, parameters=model.parameters())
    # opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())

    EPOCH_NUM = 10

    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据，变得更加简洁
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)

            # 前向计算的过程，同时拿到模型输出值和分类准确率

            # 打印模型参数和每层输出的尺寸
            predicts, acc = model(images, labels)

            # 计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)



            # 每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(),
                                                                            acc.numpy()))
                iters.append(num)
                losses.append(avg_loss)
                num += 200

            # 后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    # 保存模型参数
    paddle.save(model.state_dict(), 'mnist_test.pdparams')
    return iters, losses

class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)

        self.fc = Linear(in_features=980, out_features=10)

    # check_shape变量控制是否打印‘尺寸’,验证网络结构是否正确
    # check_content控制是否打印内容值，验证数据分布是否合理，如果中间层持续输出0，说明没有充分利用
    def forward(self, inputs, label=None, check_shape=False, check_content=False):

        output1 = self.conv1(inputs)
        output2 = F.relu(output1)
        output3 = self.max_pool1(output2)

        output4 = self.conv2(output3)
        output5 = F.relu(output4)
        output6 = self.max_pool2(output5)

        output7 = paddle.reshape(output6, [output6.shape[0], 980])
        output7 = self.fc(output7)

        # 打印神经网络每层的参数尺寸喝输出尺寸，验证网络结构是否正确
        if check_shape:
            print('print network layers superparams')
            print('conv1--kernel_size:{},padding:{},stride: {}'.format(self.conv1.weight, self.conv1._padding,
                                                                       self.conv1._stride))
            print('conv1--kernel_size:{},padding:{},stride: {}'.format(self.conv2.weight, self.conv2._padding,
                                                                       self.conv2._stride))
            print('fc-- weight_size: {}, bias_size_{}'.format(self.fc.weight.shape, self.fc.bias.shape))

            # 打印每层的输出尺寸
            print("\n########## print shape of features of every layer ###############")
            print("inputs_shape: {}".format(inputs.shape))
            print("outputs1_shape: {}".format(output1.shape))
            print("outputs2_shape: {}".format(output2.shape))
            print("outputs3_shape: {}".format(output3.shape))
            print("outputs4_shape: {}".format(output4.shape))
            print("outputs5_shape: {}".format(output5.shape))
            print("outputs6_shape: {}".format(output6.shape))
            print("outputs7_shape: {}".format(output7.shape))

        if check_content:
            # 打印卷积层的参数-卷积核权重，权重参数较多，此处只打印部分参数
            print("\n print convolution layer's kernel")
            print('conv1 params -- kernel weights:', self.conv1.weight[0][0])
            print('conv2 params -- kernel weights:', self.conv2.weight[0][0])

            # 创建随机数，随机打印某一个通道的输出值
            idx1 = np.random.randint(0, output1.shape[1])
            idx2 = np.random.randint(0, output4.shape[1])

            # 打印卷积-池化后的结果，仅打印batch中第一个图像对应的特征
            print('\n the {}th channel of conv1 layer'.format(idx1), output1[0][idx1])
            print('\n the {}th channel of conv1 layer'.format(idx2), output4[0][idx2])
            print('The output of last layer', output7[0])

        if label is not None:
            acc = paddle.metric.accuracy(input=F.softmax(output7), label=label)
            return output7, acc
        else:
            return output7


def evaluation(model):
    print('start evaluation...')

    params_path = './mnist.pdparams'
    params_dict = paddle.load(params_path)
    model.load_dict(params_dict)

    model.eval()

    eval_loader = load_data('eval')
    acc_set = []
    avg_loss_set = []

    for batch_id, data in enumerate(eval_loader()):
        imgs, labels = data
        imgs = paddle.to_tensor(imgs)
        labels = paddle.to_tensor(labels)

        predict, acc = model(imgs, labels)

        loss = F.cross_entropy(predict, labels)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

    # 计算平均损失和精确率

    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))


# 可视化分析
def draw_x_y(iter,losses):
    plt.figure()
    plt.title("train loss")
    plt.xlabel("iter")
    plt.ylabel("losses")
    plt.plot(iter, losses)
    plt.grid()
    plt.show()






if __name__ == "__main__":
    model = MNIST()
    # train_loader = load_data('train')
    # iters, losses = train(model)
    # draw_x_y(iters, losses)

    evaluation(model)

