# 导入所需要的包
import numpy as np
import json
# 导入绘图所用的包
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NetWork:

    # 初始化函数
    def __init__(self, num_of_weight):
        r = np.random.seed(0)
        self.w = np.random.rand(num_of_weight, 1)
        self.b = 0

    # 获取数据
    def load_data(self):
        # 将列名添加到读取的数据中
        list_name = ['CRIM', 'ZN', 'INDUS',
                     'CHAS', 'NOX', 'RM',
                     'AGE', 'DIS', 'RAD',
                     'TAX', 'PTRSTIO', 'B',
                     'LSTAT', 'MEDV']
        length = len(list_name)

        # 读取数据
        data = np.fromfile("./work/housing.data", sep=' ')

        # 可以当成按length个长度截一组，其中//length必须要有且不能直接写数字
        data = data.reshape([data.shape[0] // length, length])

        # 划分数据集, 由于取80%的数据作为训练集不可有0.x个数据所以要取整数进行强制转化
        ratio = 0.8

        train_length = int(data.shape[0] * ratio)

        # 通过切片进行截取
        train_data = data[:train_length]
        test_data = data[train_length:]

        # 数据归一化处理，但归一化容易被异常值影响，可使用标准化方法进行处理
        # axis表示从不同轴上进行计算，0表示沿着纵轴进行计算，1表示沿着横轴进行计算
        train_max, train_min, train_avgs = train_data.max(axis=0), train_data.min(axis=0), train_data.sum(axis=0) / \
                                           train_data.shape[0]

        # 归一化公式: (data-avgs)/(max-min)
        for i in range(length):
            train_data[:, i] = (train_data[:, i] - train_avgs[i]) / (train_max[i] - train_min[i])

        return train_data, test_data

    # 获取特征值和目标值
    def gain_x_y(self, train_data, test_data):
        # 模型设计, 目标值 = 模型（参数，特征值）
        # 获取特征值及目标值
        x = train_data[:, :-1]
        y = train_data[:, -1:]
        return x, y

    # 进行线性回归计算
    def forward(self, x):

        # 使用np.dot(x, y)，其中x为二维数组，y为一维数组进行相乘
        train_y = (np.dot(x, self.w) + self.b)

        return train_y

    # 计算损失函数
    def loss(self, z, y):

        error = z - y

        num_samples = error.shape[0]

        cost = error * error
        cost = np.mean(cost)
        cost = np.sum(cost) / num_samples
        return cost

    # 以w5，w9为例展示梯度变化图像
    def show_losses(self, x, y):
        # 当w变化时损失函数展示

        # 用来记录每个值对应的值
        losses = []
        w5 = np.arange(-160.0, 160.0, 1.0)
        w9 = np.arange(-160.0, 160.0, 1.0)
        # 重新计算损失函数得到一组结果，存放在losses中
        losses = np.zeros([len(w5), len(w9)])
        for i in range(len(w5)):

            for j in range(len(w9)):
                self.w[5] = w5[i]
                self.w[9] = w9[j]

                train_y = self.forward(x)
                predict = self.loss(train_y, y)

                losses[i, j] = predict

        fig = plt.figure()
        ax = Axes3D(fig)

        w5, w9 = np.meshgrid(w5, w9)
        ax.plot_surface(w5, w9, losses, rstride=1, cstride=1, cmap="rainbow")
        plt.show()

    # 计算w与b梯度过程
    def gradient(self, x, y, train_y):

        # 计算梯度
        # L = 1/2(y-z)^2（梯度计算公式）
        # 偏L/偏w、偏L/偏b，w(i = 0, 1, 2,...,12)

        """
        一种实现方法，通过for循环来做（下列只计算了第一个样本的w，b）
        gradient_w = []
        for i in range(13):
            gradient_w.extend((y_trian[i] - y[i]) * x[i][i])
        print("grandient_的值为：", gradient_w)
        gradient_b = y_train[0] - y[0]
        """

        # 通过numpy自带的直接计算
        N = x.shape[0]
        gradient_w = 1.0 / N * np.sum((y - train_y) * x, axis=0)
        """
        gradient_w = (y - train_y) * x

        # 将计算出的wi进行相加求平均
        gradient_w = np.mean(gradient_w, axis = 0)

        # w.shape为(13,),gradient_w的形状为（404,13）,为了与gradient_w的形状一致进行计算
        # 对w进行变形
        # np.newaxis是指加入一个新维度
        """

        gradient_w = gradient_w[:, np.newaxis]

        """
        # 计算偏置b 偏L/偏b = y - z, 其中y为真实值, z为预测值
        gradient_b = y - train_y

        # 计算grandient_b的平均值
        gradient_b = np.mean(gradient_b)
        """

        gradient_b = 1.0 / N * np.sum(y - train_y)

        return gradient_w, gradient_b

    # 迭代计算参数
    def train(self, x, y, train_y, iterations, eta=0.1):
        losses = []

        # 计算损失函数
        for i in range(iterations):
            train_y = self.forward(x)
            predict = self.loss(train_y, y)

            losses.append(predict)
            gradient_w, grandient_b = self.gradient(x, y, train_y)

            self.w = self.w - eta * gradient_w
            self.b = self.b - eta * grandient_b

            return losses

    # 随机计算损失函数
    def random_train(self, train_data, num_epochs, len_size=10, eta=0.1):
        """
        param:train_data是训练数据
        param:iterations是迭代次数
        param:len_size是每次取出多少数据
        param:num_epochs是训练次数
        param:是步长
        """
        n = len(train_data)
        losses = []

        for epoch_id in range(num_epochs):
            # 首先将样本打乱顺序，为了避免模型记忆影响训练效果
            np.random.shuffle(train_data)
            mini_batchs = [train_data[k:k + len_size] for k in range(0, n, len_size)]

            # 每次按比例取出样本作为mini-batch
            # enumerate返回序列和元素
            for iter_id, mini_batch in enumerate(mini_batchs):
                x1 = mini_batch[:, :-1]
                y1 = mini_batch[:, -1:]

                train_y1 = self.forward(x1)
                predict = self.loss(train_y1, y1)

                gradient_w, grandient_b = self.gradient(x1, y1, train_y1)

                self.w = self.w - eta * gradient_w
                self.b = self.b - eta * grandient_b

                losses.append(predict)

                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.format(epoch_id, iter_id, predict))

        return losses


if __name__ == "__main__":
    network = NetWork(13)

    # 1. 获取数据
    train_data, test_data = network.load_data()

    # 2. 获取特征值和目标值
    x, y = network.gain_x_y(train_data, test_data)

    # 3. 进行线性回归计算
    train_y = network.forward(x)

    # 4. 损失函数计算
    predict = network.loss(train_y, y)

    # 5. 梯度计算
    gradient_w, grandient_b = network.gradient(x, y, train_y)

    # 6. 确定损失函数最小点
    # network.train(x, y, train_y, 200, eta=0.1)

    # 7. 随机梯度下降法
    losses = network.random_train(train_data, 50, len_size=100)

    # 画出损失函数的变化趋势
    plot_x = np.arange(len(losses))
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()












