import paddle.fluid
from paddle.nn import Conv2D, MaxPool2D, Linear
"""

卷积：
    卷积是数学分析中得一种积分变换方法，在卷积层中是互相关运算
    卷积核：也被称作滤波器，
    卷积核的计算过程可以用下面的数学公式表示，其中a代表输入图片，b代表输出特征图，w是卷积核参数，它们都是二维数组，∑u,v表示对卷积核参数进行遍历并求和。

    b[i,j]=∑u,v(a[i+u,j+v]⋅w[u,v])，也可在剪辑和中加入偏置操作

填充：
    若输入得图片得格式是3*3，卷积核大小为2*2，经过一次卷积后，图片尺寸变小
    卷积输出特征图得尺寸计算方法：hv = Hv - kv +1; hw = Hw - kw - 1
    其中，kv, kw分别是卷积核得长和宽

步幅：
    卷积核在图片上移动式得移动像素个数

感受野：
    输出特征图上每个点得数值是由图片上大小为kh，kw得区域元素与卷积核每个元素相乘再相加得到
    输入图像上得kh，kw区域每个元素值得改变都会影响输出得像素值，这个区域就叫感受野

池化：
    是将某一位置相邻输出的总体统计特征代替网络在该位置的输出
    通常使用平均池化和最大池化
    优点：输入数据做出少量平移时，经过池化函数后的大多数输出值能保持不变
    在卷积网络中使用的池化窗口大多是2*2，步长也为2，填充为0

激活函数Relu：
    y =0, x<0; y = x, x>0

激活函数simgmoid:
    y = 1/ 1 + e^-x

梯度消失现象
    在神经网络里，将经过反向传播之后，梯度值衰减到接近于零的现象称作梯度消失现象。

批归一化：
    # 使用BatchNorm进行归一化输出
    # 输入维度为[N, K]，num_features = K
    # 输入数据结构是[N,C,H,W]时，一般对应卷积层的输出，一般沿着C这一维度展开
    # 分别计算每个通道中N*H*W个像素点的均值和方差
    在预测时使用BatchNorm预测结果可能会出现不确定性，在BatchNorm中会采用计算均值和方差的移动平均值

丢弃法：
    用来抑制过拟合，随机选择一部分神经元设置其输出为0，
    Dropout(丢弃法)的两种解决方案，
    1，downgrade_in_infer：训练时随机丢弃一部分神经元，预测时不丢弃神经元，但把它们的数值变小】
    2.upscale_in_train：训练时随机丢弃一部分神经元，但是把保留的哪些神经元数值放大，预测时原样输出所有神经元的信息
    
    
    
    
    
"""
conv = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)

# 查看卷积层参数
with paddle.fluid.dygraph.guard():
    # 通过conv.parameters()查看卷积层的参数，返回值是list，
    print(conv.parameters())
    # 查看卷积层的权重参数名字和数值
    print(conv.parameters()[0].name, conv.parameters()[0].numpy())
    # 查看卷积层的偏置参数名字和数值
    print(conv.parameters()[1].name, conv.parameters()[1].numpy())
