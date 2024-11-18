import paddle
import random
import json
from paddle.nn import Conv2D, MaxPool2D, Linear, BatchNorm1D, BatchNorm2D, Dropout
import numpy as np
import matplotlib.pyplot as plt
from paddle.nn.initializer import Assign
from PIL import Image

# 黑白边缘检测
def black_wite():

    # 设置初始化权重参数
    w = np.array([1, 0, -1], dtype='float32')
    # 将权重参数转为[cout, cin, kh, kw]的四维张量
    w = w.reshape([1, 1, 1, 3])

    # 创建卷积算子的时候，通过参数属性weight_attr指定参数初始化方式
    # 这里的初始化方式是从numpy.ndarray初始化卷积参数
    conv = Conv2D(in_channels=1, out_channels=1, kernel_size=[1, 3],
                  weight_attr=paddle.ParamAttr(initializer=Assign(value=w)))

    img = np.ones([50, 50], dtype='float32')
    img[:, 30:] = 0

    # 将土图片调整为[N, C, H, W]
    x = img.reshape([1, 1, 50, 50])

    x = paddle.to_tensor(x)
    y = conv(x)
    output = y.numpy()

    f = plt.subplot(121)
    f.set_title('input image', fontsize=15)
    plt.imshow(img, cmap='gray')

    f = plt.subplot(122)
    f.set_title('output featuremap', fontsize=15)
    plt.imshow(output.squeeze(), cmap='gray')
    plt.show()

# 图片边缘检测
def bianyuan():

    img = Image.open('tyh_1.jpg')


    w = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    w = w.reshape([1, 1, 3, 3])

    # 由于输入通道为3，所以将卷积核设置为[1, 3, 3, 3]
    w = np.repeat(w, 3, axis=1)

    conv = Conv2D(in_channels=3, out_channels=1, kernel_size=[3, 3],
                  weight_attr=paddle.ParamAttr(initializer=Assign(value=w)))

    # 将图片转为numpy.ndarray
    x = np.array(img).astype('float32')
    # 图片读入ndarray时，形状时[H, W, 3]
    # 将通道维度调整到最前面
    x = np.transpose(x, (2, 0, 1))

    # 将图片调整为[N, C, H, W]格式
    x = x.reshape(1, 3, img.height, img.width)
    x = paddle.to_tensor(x)

    y = conv(x)
    output = y.numpy()

    plt.figure(figsize=(20, 10))
    f = plt.subplot(121)
    f.set_title('input image', fontsize=15)
    plt.imshow(img)
    f = plt.subplot(122)
    f.set_title('output feature map', fontsize=15)
    plt.imshow(output.squeeze(), cmap='gray')
    plt.show()

# 均值模糊
def avgs_mohu():

    img = Image.open('tyh_1.jpg').convert('L')
    img = np.array(img)
    w = np.ones([1, 1, 5, 5]).astype('float32')

    conv = Conv2D(in_channels=3, out_channels=1, kernel_size=[5, 5],
                  weight_attr=paddle.ParamAttr(initializer=Assign(value=w)))

    x = img.astype('float32')
    x = x.reshape([1, 1, img.shape[0], img.shape[1]])
    x = paddle.to_tensor(x)

    y = conv(x)
    output = y.numpy()

    plt.figure(figsize=(20, 12))
    f = plt.subplot(121)
    f.set_title('input image')
    plt.imshow(img, cmap='gray')

    f = plt.subplot(122)
    f.set_title('output feature map')
    out = output.squeeze()
    plt.imshow(out, cmap='gray')

    plt.show()


# 批归一化
def BatchNorm():

    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype('float32')
    # 使用BatchNorm进行归一化输出
    # 输入维度为[N, K]，num_features = K
    bn = BatchNorm1D(num_features=3)
    x = paddle.to_tensor(data)
    y = bn(x)
    print('output of BatchNorm Layer:{}'.format(y.numpy()))


    # 输入数据结构是[N,C,H,W]时，一般对应卷积层的输出，一般沿着C这一维度展开
    # 分别计算每个通道中N*H*W个像素点的均值和方差
    np.random.seed(100)
    data = np.random.rand(2, 3, 3, 3).astype('float32')
    print(data.shape)
    # 使用BatchNorm计算归一化输出
    # 输入维度为[N,C,H,W]，num_feature = C
    bn_1 = BatchNorm2D(num_features=3)
    x_1 = paddle.to_tensor(data)
    y_1 = bn_1(x_1)
    print('input of BatchNorm2D Layer: {}'.format(x_1.numpy()))
    print('output of BatchNorm2D Layer: {}'.format(y_1.numpy()))


# 丢弃法，用来抑制过拟合
# 1，downgrade_in_infer：训练时随机丢弃一部分神经元，预测时不丢弃神经元，但把它们的数值变小】
# 2.upscale_in_train：训练时随机丢弃一部分神经元，但是把保留的哪些神经元数值放大，预测时原样输出所有神经元的信息
def dropout():

    np.random.seed(100)
    # 对应卷积层的输出,即中间输出的部分
    data = np.random.rand(2, 3, 3, 3).astype('float32')
    # 对应全连接层的输出,即最后输出的部分
    data_2 = np.arange(1, 13).reshape([-1, 3]).astype('float32')

    x_1 = paddle.to_tensor(data)
    x_2 = paddle.to_tensor(data_2)

    drop_1 = Dropout(p=0.5, mode='downscale_in_infer')
    drop1_trian_x1 = drop_1(x_1)
    drop1_trian_x2 = drop_1(x_2)
    # 切换到eval模式
    drop_1.eval()
    drop1_eval_x1 = drop_1(x_1)
    drop1_eval_x2 = drop_1(x_2)

    drop_2 = Dropout(p=0.5, mode='upscale_in_train')
    drop2_train_x1 = drop_2(x_1)
    drop2_train_x2 = drop_2(x_2)
    # 切换到eval模式
    drop_2.eval()
    drop2_eval_x1 = drop_2(x_1)
    drop2_eval_x2 = drop_2(x_2)

    print('drop1_train_x1: {}, drop1_train_x2: {}, drop1_eval_x1: {}, drop1_eval_x2: {}'.format(drop1_trian_x1,
                                                                                                drop1_trian_x2,
                                                                                                drop1_eval_x1,
                                                                                                drop1_eval_x2))
    print('drop2_train_x1: {}, drop2_train_x2: {}, drop2_eval_x1: {}, drop2_eval_x2: {}'.format(drop2_train_x1,
                                                                                                drop2_train_x2,
                                                                                                drop2_eval_x1,
                                                                                                drop2_eval_x2))






if __name__ == "__main__":

    # black_wite()
    # bianyuan()
    # avgs_mohu()
    # BatchNorm()
    dropout()


