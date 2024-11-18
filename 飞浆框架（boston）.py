import numpy

import paddle
import numpy as np
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import random

mnist = np.fromfile('new.txt', sep=",")
print(mnist)
print(len(mnist))
print(max(mnist))

def load_data():

    list_name = ['CRIM', 'ZN', 'INDUS',
                 'CHAS', 'NOX', 'RM',
                 'AGE', 'DIS', 'RAD',
                 'TAX', 'PTRSTIO', 'B',
                 'LSTAT', 'MEDV']
    length = len(list_name)

    data = np.fromfile("./housing.data", sep=" ")

    # 重新划分数据
    data = data.reshape([data.shape[0] // length, length])

    print(data.shape[0])

    # 归一化处理
    data_max = data.max(axis=0)
    data_min = data.min(axis=0)
    data_avgs = data.sum(axis=0) / data.shape[0]
    global max_data
    global min_data
    global avgs_data
    max_data = data_max
    min_data = data_min
    avgs_data = data_avgs
    for k in range(length):
        data[:, k] = (data[:, k] - data_avgs[k]) / (data_max[k] - data_min[k])

    # 拆分训练集和目标集
    split_len = int(data.shape[0] * 0.8)
    train_data = data[:split_len]
    test_data = data[split_len:]
    print(train_data.shape)

    # 返回训练集和目标集
    return train_data, test_data


# 拆分特征值和目标值
def train_x_y(train_data, test_data):

    x_train = train_data[:, :-1]
    y_train = train_data[:, -1:]
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1:]

    return x_train, y_train, x_test, y_test

# 定义线性回归网络结构，需要继承paddle.nn.Layer父类
class Regressor(paddle.nn.Layer):
    def __init__(self):
        super(Regressor, self).__init__()

        # 定义一层全连接层，输入的维度是13，输出维度为1
        self.fc = Linear(in_features=13, out_features=1)

    # 向前计算
    def forward(self, inputs):
        z = self.fc(inputs)

        return z

# 训练配置过程
# 1. 指定运行训练的机器资源
# 2. 声明模型实例
# 3. 加载训练和测试数据
# 4. 设置优化算法和学习率


if __name__ == "__main__":

    train_data, test_data = load_data()
    # 1.指定训练的机器资源,默认使用AI Studio模型，本次不指定
    # 2.声明模型实例
    model = Regressor()

    # 3.加载训练模型和测试数据
    x_train, y_train, x_test, y_test = train_x_y(train_data, test_data)
    model.train()   # 训练状态train(), 预测状态eval()

    # 4. 设置优化算法和学习率
    opt = paddle.optimizer.SGD(learning_rate=0.02, parameters=model.parameters())

    # 进行训练
    EPOCH_NUM = 10  # 外层循环数
    BATCH_SIZE = 10    # 拆分的个数
    for epoch_id in range(EPOCH_NUM):
        np.random.shuffle(train_data)

        # 对其进行拆分
        mini_batches = [train_data[k:k+BATCH_SIZE] for k in range(0, len(train_data), BATCH_SIZE)]

        # 根据索引进行计算
        for iter_id, mini_batch in enumerate(mini_batches):
            x = np.array(mini_batch[:, :-1]).astype(np.float32)
            y = np.array(mini_batch[:, -1:]).astype(np.float32)

            # 将numpy数据转为飞浆的tensor动态图格式
            """
            静态图模式（声明式编程模式，C++）先编译后执行，性能好，便于部署
            
            动态图模式（命令式编程范式，Python）解析式执行方式
            
            """
            house_features = paddle.to_tensor(x)
            prices = paddle.to_tensor(y)

            # 前向计算
            predict = model(house_features)

            # 计算损失函数
            loss = F.square_error_cost(predict, label=prices)

            # 计算均值
            avg_loss = paddle.mean(loss)

            if(iter_id % 20 == 0):
                print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))

            # 反向传播，计算每层参数的梯度值
            avg_loss.backward()

            # 更新参数，进行迭代
            opt.step()

            # 清空梯度变量，进行下一轮计算
            opt.clear_grad()

    # 保存模型参数，文件名为LR_model.pdparams
    paddle.save(model.state_dict(), 'LR_model.pdparams')
    print("模型保存成功，模型参数保存在LR_model.pdparams中")

    # 进行测试
    model_dict = paddle.load('LR_model.pdparams')
    model.load_dict(model_dict)
    model.eval()
    print(x_test.shape)

    x_test = np.array(x_test).astype(np.float32)

    # 将测试数据转为动态图
    tensor_test_data = paddle.to_tensor(x_test)



    # 进行预测
    predict = model(tensor_test_data)

    # 反归一化处理
    predict = predict * (max_data[-1] - min_data[-1]) + avgs_data[-1]

    y_test = y_test * (max_data[-1] - min_data[-1]) + avgs_data[-1]

    print("Inference result is {}, the corresponding label is {}".format(predict.numpy(), y_test))




