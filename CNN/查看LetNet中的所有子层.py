import paddle
import paddle.nn.functional as F
import numpy as np
import random
from paddle.nn import Conv2D, MaxPool2D, Linear

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


x = np.random.randn(*[3, 1, 28, 28])
x = x.astype('float32')
model = LeNet1()
# 使用LeNet从基类中继承的sublayers()函数查看LeNet中包含的子层
print("所包含的子层是", model.sublayers())

x = paddle.to_tensor(x)
for item in model.sublayers():

    try:
        x = item(x)
    except:
        x = paddle.reshape(x, [x.shape[0], -1])
        x = item(x)
    if len(item.parameters()) == 2:
        # 查看卷积和全连接层的数据和参数的形状，
        # 其中item.parameters()[0]是权重参数w，item.parameters()[1]是偏置参数b
        print(item.full_name(), x.shape, item.parameters()[0].shape, item.parameters()[1].shape)
    else:
        # 池化层没有参数
        print(item.full_name(), x.shape)
