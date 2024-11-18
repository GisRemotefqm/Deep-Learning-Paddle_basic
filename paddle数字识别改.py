import paddle
import os
import numpy as np
import paddle.nn.functional as F
import json
import random
from paddle.nn import Linear
import gzip
from paddle.io import Dataset


print(paddle.device.get_device())
paddle.device.set_device('gpu:0')

class MNIST(paddle.nn.Layer):

    def __init__(self):
        super(MNIST, self).__init__()

        # 定义全连接层
        self.fc = Linear(in_features=784, out_features=1)

    def forward(self, inputs):
        output = self.fc(inputs)

        return output


# 加载数据
def load_data(model='train'):
    file_path = './mnist.json.gz'
    data = json.load(gzip.open(file_path))

    train_data, val_data, test_data = data

    train_img = train_data[0]
    train_label = train_data[1]

    val_img = val_data[0]
    val_label = val_data[1]

    test_img = test_data[0]
    test_label = test_data[1]

    if model == 'train':

        return train_img, train_label

    elif model == 'eval':

        return test_img, test_label

    elif model == 'valid':

        return val_img, val_label

    else:

        raise Exception("mode can only be one of ['train', 'valid', 'eval']")


# 对数据进行分组
def group_data():

    # 加载数据
    train_data, train_label = load_data(model="train")

    # 打乱数据,由于数据是列表形式所以通过序号进行打乱
    img_length = len(train_data)
    index_list = list(range(img_length))
    random.shuffle(index_list)

    # for循环进行训练

    BATCHSIZE = 100

    imgs_list = []
    label_list = []

    # 第二层循环获得数据
    for i in index_list:
        img = np.array(train_data[i]).astype('float32')
        label = np.array(train_label[i]).astype('float32')
        imgs_list.append(img)
        label_list.append(label)

        if len(imgs_list) == BATCHSIZE:
            yield np.array(imgs_list), np.array(label_list)

            imgs_list = []
            label_list = []

        if len(imgs_list) > 0:
            yield imgs_list, label_list

    # 相当于第一层循环，将数据重新加载打乱
    return group_data


# 训练配置，启动训练过程
def trian(model, data_loader):

    model.train()

    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())

    EPOCH_NUM = 10
    for epoch_id in range(EPOCH_NUM):

        # 声明函数
        # train_loader = group_data
        for iter_id, data in enumerate(data_loader()):
            imgs_list, label_list = data
            print("img_list", imgs_list.shape)
            # 转为tensor
            tensor_img = paddle.to_tensor(imgs_list)
            tensor_label = paddle.to_tensor(label_list)

            # 前向计算
            predict = model(tensor_img)

            # 计算损失函数
            loss = F.square_error_cost(predict, tensor_label)
            avgs_loss = paddle.mean(loss)

            if iter_id % 200 == 0:
                print("epoch:{}, batch:{}, loss:{}".format(epoch_id, iter_id, loss))

            # 后向计算
            avgs_loss.backward()
            opt.step()
            opt.clear_grad()

    # 保存模型
    paddle.save(model.state_dict(), "./minist1.pdparams")


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



if __name__ == "__main__":

    model = MNIST()

    # trian(model)

    # 使用异步方式加载数据
    train_dataset = MnistDataset('train')

    data_loder = paddle.io.DataLoader(train_dataset, batch_size=100, shuffle=True)

    trian(model, data_loder)
