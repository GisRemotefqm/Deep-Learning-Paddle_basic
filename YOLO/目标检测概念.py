"""

目标检测的主要目的是让计算机自动识别图片或视频中目标的类别
并且在目标周围绘制边界框

检测方法：
    使用某种方式在输入图片上生成一系列可能包含物体的区域，然后对
    某个候选区当成一幅画像看待，再使用图像分类模型进行分类
    穷举法产生的数目大约为 W^2 * H^2 / 4

通常使用两种方法表示边界框的位置
    1. (x1, y1, x2, y2), 左上右下
    2. (x, y, w, h), 中心点坐标和宽、高

在目标检测中训练集的标签会给出物体真实边界的(x1, y1, x2, y2)

锚框：
    锚框是一种假想框。先设定锚框的大小和形状，再在图像上某一中心画框

交并比：
    使用了集合的概念真实框A与锚框B，A交B / A并B

检测任务输出：[L, P, x1, y1, x2, y2], L类别标签，P为是此标签的概率

图像增广：
    主要作用是为了扩大训练数据集，抑制过拟合，提升模型泛化能力
    常用的方法：随机改变亮暗、对比度、颜色等
              随机填充、裁剪、缩放、翻转

2013年，Ross Girshick 等人于首次将CNN的方法应用在目标检测任务上，
他们使用传统图像算法Selective Search产生候选区域，取得了极大的成功，
这就是对目标检测领域影响深远的区域卷积神经网络(R-CNN)模型。

2015年，Ross Girshick 对此方法进行了改进，提出了Fast R-CNN模型。
通过将不同区域的物体共用卷积层的计算，大大缩减了计算量，提高了处理速度，
而且还引入了调整目标物体位置的回归方法，进一步提高了位置预测的准确性。

2015年，Shaoqing Ren 等人提出了Faster R-CNN模型，提出了RPN的方法来产生物体的候选区域，
这一方法不再需要使用传统的图像处理算法来产生候选区域，进一步提升了处理速度。

2017年，Kaiming He 等人提出了Mask R-CNN模型，
只需要在Faster R-CNN模型上添加比较少的计算量，
就可以同时实现目标检测和物体实例分割两个任务。

每个锚框需要标注数据的个数
location(4) + objectness(1) + classification(C): 5 + C
m * n个方块区域，每个区域都有K个锚框，则标注数据的维度是[K(5+C)] * m * n

"""



import numpy as np


# 计算交并比
def bor_iou_xy(box_1, box_2):

    x_1min, y_1min, x_1max, y_1max = box_1[0], box_1[1], box_1[2], box_1[3]
    s1 = (y_1max - y_1min + 1.) * (x_1max - x_1min + 1.)

    x_2min, y_2min, x_2max, y_2max = box_2[0], box_2[1], box_2[2], box_2[3]
    s2 = (y_2max - y_2min + 1.) * (x_2max - x_2min + 1.)

    # 寻找相交坐标值
    xmin = np.maximum(x_1min, x_2min)
    xmax = np.minimum(x_1max, x_2max)
    ymin = np.maximum(y_1min, y_2min)
    ymax = np.minimum(y_1max, y_2max)

    # 计算相交的面积
    inter_h = np.maximum(ymax - ymin + 1., 0.)
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    intersetion = inter_w * inter_h

    # 计算相并面积
    union = s1 + s2 - intersetion

    # 计算相交比
    iou = intersetion / union

    return iou


if __name__ == "__main__":
    bbox1 = [100., 100., 200., 200.]
    bbox2 = [120., 120., 220., 220.]
    iou = bor_iou_xy(bbox1, bbox2)
    print('IoU is {}'.format(iou))
