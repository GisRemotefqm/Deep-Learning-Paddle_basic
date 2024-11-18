import numpy as np
import matplotlib.pyplot as plt
import json

def load_data():
    # 将列名添加到读取的数据中
    list_name = ['CRIM', 'ZN', 'INDUS',
                 'CHAS', 'NOX', 'RM',
                 'AGE', 'DIS', 'RAD',
                 'TAX', 'PTRSTIO', 'B',
                 'LSTAT', 'MEDV']
    length = len(list_name)

    # 读取数据
    data = np.fromfile("./housing.data", sep=' ')

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

class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        # np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.

    def forward(self, x):
        z = np.dot(x, self.w) + self.b

        return z

    def loss(self, z, y):
        error = z - y

        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost

    def gradient(self, x, y):
        z = self.forward(x)
        N = x.shape[0]

        gradient_w = 1. / N * np.sum((z - y) * x, axis=0)
        gradient_w = gradient_w[:, np.newaxis]


        gradient_b = 1. / N * np.sum(z - y)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    def train(self, training_data, num_epochs, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epochs):
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                # print(self.w.shape)
                # print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                print("def train x.shape = ", x.shape)
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                      format(epoch_id, iter_id, loss))

        return losses


# 获取数据
train_data, test_data = load_data()

# 创建网络
net = Network(13)
# 启动训练
losses = net.train(train_data, num_epochs=50, batch_size=100, eta=0.1)

# 画出损失函数的变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
