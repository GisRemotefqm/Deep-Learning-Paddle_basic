"""
经典全连接神经网络：
    包括：输入层，两个隐含层，输出层
输入层：将数据输入到神经网络中
隐含层：增加网络的深度和复杂度，隐含层的节点数是可以调整的，节点越多越复杂，能力越强，参数量也会随之增加，激活函数常见的是Sigmoid函数
输出层：输出结果


卷积神经网络：
    卷积神经网络对视觉问题的特点进行了网络结构优化，保留了像素间的空间信息，保留关键的特征信息
    卷积神经网络是由多个卷积层和池化层组成
    卷积层负责对输入进行扫描生成更抽象的特征
    池化层表示对特征进行过滤

"""
from sklearn.model_selection import train_test_split
import os
import paddle
import random
from paddle.nn import Linear, Conv2D, MaxPool2D
import paddle.nn.functional as F
import numpy as np
import json
import gzip
from paddle.io import Dataset
import pandas as pd


class MNIST(paddle.nn.Layer):

    def __init__(self):
        super(MNIST, self).__init__()

        # 定义两个隐含层
        # 开始输入
        self.fc = Linear(in_features=784, out_features=10)
        # 隐含层
        self.fc_2 = Linear(in_features=10, out_features=10)
        self.fc_3 = Linear(in_features=10, out_features=1)

    def forward(self, inputs):

        # 定义激活函数, 隐含层定义激活函数，输出不用
        output = self.fc(inputs)
        output = F.relu(output)
        output_2 = self.fc_2(output)
        output_2 = F.relu(output_2)
        output_3 = self.fc_3(output_2)

        return output_3



def load_data(mode = 'train'):

    file_path = './mnist.json.gz'
    data = json.load(gzip.open(file_path))
    train_data, valid_data, evla_data = data

    if mode == 'train':
        img = train_data[0]
        label = train_data[1]

        return img, label
    elif mode == 'eval':
        img = evla_data[0]
        label = evla_data[1]

        return img, label
    elif mode == 'valid':
        img = valid_data[0]
        label = valid_data[1]

        return img, label
    else:
        raise Exception("mode = 'train', 'eval', 'valid")


def bm_load_data(mode='train'):

    csvPath = r"J:\发送文件\Atmosphere\TrainData\230128_TrainData.csv"
    csvPath2 = r"J:\发送文件\Atmosphere\TrainData\231031_TrainData.csv"
    df1 = pd.read_csv(csvPath)
    df2 = pd.read_csv(csvPath2)

    df1_feature1 = df1['area', 'mojor_length', 'perimeter', 'area_convex', 'monor_length', 'extent', 'convex_area', 'blue_mean',
              'green_mean', 'red_mean', 'nir_mean', 'blue_std', 'green_std', 'red_std', 'nir_std', 'blue_max', 'green_max',
                'red_max', 'nir_max', 'blue_min', 'green_min', 'red_min', 'nir_min', 'ndvi_mean', 'ndvi_std', 'ndvi_max',
              'ndvi_min','ndwi_mean', 'ndwi_std', 'ndwi_max', 'ndwi_min', 'kt1_mean', 'kt1_min', 'kt1_max', 'kt1_std',
              'glcm_meant', 'glcm_mean_std', 'glcm_mean_max', 'glcm_mean_min']

    df2_feature2 = df2['area', 'mojor_length', 'perimeter', 'area_convex', 'monor_length', 'extent', 'convex_area', 'blue_mean',
              'green_mean', 'red_mean', 'nir_mean', 'blue_std', 'green_std', 'red_std', 'nir_std', 'blue_max', 'green_max',
              'red_max', 'nir_max', 'blue_min', 'green_min', 'red_min', 'nir_min', 'ndvi_mean', 'ndvi_std', 'ndvi_max',
              'ndvi_min', 'ndwi_mean', 'ndwi_std', 'ndwi_max', 'ndwi_min', 'kt1_mean', 'kt1_min', 'kt1_max', 'kt1_std',
              'glcm_meant', 'glcm_mean_std', 'glcm_mean_max', 'glcm_mean_min']

    df1_label = df1['classfiterLabel']
    df2_label = df2['classfiterLabel']

    Allfeatures = df1_feature1.to_numpy().extend(df2_feature2.to_numpy())
    AllLabel = df1_label.to_numpy().extend(df2_label.to_numpy())

    if mode == 'train':
        pass
    elif mode == 'vaild':
        pass
    else:
        raise Exception("mode = 'train', 'eval', 'valid")

    return feature, label



# 数据生成器
def data_generator():

    imgs, labels = load_data()

    # 通过索引打乱顺序
    length = len(imgs)
    index_list = list(range(length))
    random.shuffle(index_list)

    # 获得数据
    BATCH_SIZE = 100
    img_list = []
    label_list = []
    img_r = 28
    img_h = 28

    for i in index_list:


        img = np.array(imgs[i]).astype('float32')
        label = np.array(labels[i]).astype('float32')

        # 卷积神经网络需要图像信息所以需要将其改为28 * 28
        img = np.reshape(imgs[i], [1, img_r, img_h]).astype('float32')
        print(img.shape)
        label = np.reshape(labels[i], [1]).astype('float32')

        img_list.append(img)
        label_list.append(label)

        if len(img_list) == BATCH_SIZE:

            yield np.array(img_list), np.array(label_list)
            img_list = []
            label_list = []

    # if len(img_list) > 0:
    #
    #     yield np.array(img_list), np.array(label_list)

    return data_generator

# 训练过程
def train(model):

    model.train()

    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())

    EPOCH_NUM = 10
    for epoch_id in range(EPOCH_NUM):

        generator = data_generator
        for iter_id, data in enumerate(generator()):

            # 转为动态图
            img_list, label_list = data
            # print("img_list", img_list.shape)
            tensor_img = paddle.to_tensor(img_list)
            tensor_label = paddle.to_tensor(label_list)

            # 向前计算
            predict = model(tensor_img)

            # 计算损失
            loss = F.square_error_cost(predict, tensor_label)
            avgs_loss = paddle.mean(loss)

            if iter_id % 200 == 0:

                print("epoch:{}, batch:{}, loss:{}".format(epoch_id, iter_id, avgs_loss))

            # 向后计算
            avgs_loss.backward()
            opt.step()
            opt.clear_grad()

    # 保存模型
    paddle.save(model.state_dict(), './mnist2.pdparams')


# 异步读取数据
class MnistDataset(Dataset):

    def __init__(self, model):

        data_file = "./mnist.json.gz"
        data = json.load(gzip.open(data_file))

        train_data, val_data, eval_data = data

        if model == 'train':

            img = train_data[0]
            label = train_data[1]

        elif model == 'eval':

            img = eval_data[0]
            label = eval_data[1]

        elif model == 'valid':

            img = val_data[0]
            label = val_data[1]

        else:
            raise Exception("mode can only be one of ['train', 'valid', 'eval']")

        self.img = img
        self.label = label

        def __getitem__(self, item):

            img = np.array(self.img[item]).astype('float32')
            label = np.array(self.label[item]).astype('float32')

            return img, label

        def __len__(self):

            return len(self.img)


class MNIST_2(paddle.nn.Layer):

    def __init__(self):
        super(MNIST_2, self).__init__()

        # 定义卷积层
        # 输出特征通道为20，卷积核大小为5，卷积步长为1，padding=2
        self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层,池化核大小为2，池化步长为2
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 定义卷积层
        self.conv2 = Conv2D(in_channels=20,out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)

        # 定义全连接层
        self.fc = Linear(in_features=980, out_features=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x

if __name__ == '__main__':

    mnist = MNIST_2()
    train(mnist)



