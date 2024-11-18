import paddle
import numpy as np
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# 确保从datasets中的数据类型都为np.narray类型
paddle.vision.set_image_backend('cv2')


# 加载数据
def load_data():
    # 读取出来是图像类型
    train_data = paddle.vision.datasets.MNIST(mode="train")
    train_data = np.array(train_data)

    # 数组中第一位存的是图片，第二位是标签
    # 图片的大小是28*28，所以一共有784个维度，但由于有空间位置关系所以是不可变的所以要明确图像内容

    # 转为数组
    print(np.array(train_data[2][0]))

    return train_data


class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        # 定义全连接层
        self.fc = paddle.nn.Linear(in_features=784, out_features=1)

    def forward(self, inputs):
        output = self.fc(inputs)

        return output

    # 设置优化算法和学习率


def train(model):
    # 设置训练模式
    model.train()

    # 加载训练集
    train_data = paddle.io.DataLoader(paddle.vision.MNIST(mode="train"), batch_size=50, shuffle=True)

    print("train_data", train_data)

    # 设置优化算法和学习率
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())

    # 进行循环优化参数
    EPOCH_NUM = 10
    for epoch in range(EPOCH_NUM):

        for data_id, data in enumerate(train_data()):

            image = norm_img((data[0]).astype('float32'))
            label = data[1].astype('float32')

            # 向前计算
            predict = model(image)

            # 计算损失
            loss = F.square_error_cost(predict, label)
            avgs_loss = paddle.mean(loss)

            if data_id % 1000 == 0:
                print("epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, data_id, avgs_loss.numpy()))

            # 向后计算
            avgs_loss.backward()
            opt.step()
            opt.clear_grad()

    paddle.save(model.state_dict(), './mnist.pdparams')


# 对影像进行归一化处理
def norm_img(img):
    assert len(img.shape) == 3, "img不是三维的"

    # print("img为", img)
    batch_size, img_h, img_w = img.shape[0], img.shape[1], img.shape[2]
    # print("batch_size:{},img_h:{},img_w:{}".format(batch_size, img_h, img_w))

    # 对数据进行归一化处理
    img = img / 255

    img = paddle.reshape(img, [batch_size, img_w * img_h])

    return img


if __name__ == "__main__":
    data = load_data()

    # 生成模型实例
    model = MNIST()

    train(model)

    # 模型测试
    test_x = Image.open('./example_6.jpg')

    plt.imshow(test_x)
    plt.show()

    # 将图像转为灰度图
    test_x = test_x.convert('L')

    # 使用Image.ANTALIAS方式对图像进行重采样
    test_x = test_x.resize((28, 28), Image.ANTIALIAS)
    print("重采样后的图片", test_x)
    test_x = np.array(test_x).reshape(1, -1).astype(np.float32)
    print("reshape之后的test_x.shape", test_x)

    # 对图像进行归一化处理
    test_x = 1 - test_x / 255

    # 导入模型
    param_dict = paddle.load('mnist.pdparams')
    model.load_dict(param_dict)
    model.eval()

    result = model(paddle.to_tensor(test_x))

    print("本次预测的数字是", result.numpy().astype('int32'))