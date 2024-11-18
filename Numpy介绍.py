import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
c = a + b
print(c)

c = a + 1
print(c)

# numpy的广播机制
d = np.array([[1, 2, 3, 4], [11, 22, 33, 44]])
print(d.shape)
c = d + a
print(c)

# array, arange, zeros, ones函数的用法

# array生成数组
arr = [[[1, 2, 3, 4],[4, 5, 6, 7], [8, 9, 10, 11]], [[12, 5, 8, 6],[7, 3, 5, 1],[5, 5, 5, 5]]]

arr = np.array(arr)
print(arr)
print("arr1", arr[1])

# arange创建从元素10依次递增2的数组
arr = np.arange(0, 10, 2)
print("依次递增", arr)

# 创建指定长度或者形状的全0数组
arr = np.zeros([3, 4])
print("全零数组", arr)

# 创建指定长度或者形状的全1数组
arr = np.ones([4, 5, 3])
print(arr)

# 查看数组属性 shape, dtype, size, ndim

# shape查看数组形状
print("查看数组形状", arr.shape)

# 查看数据类型
print("查看数据类型", arr.dtype)

# 数组中包含的元素个数，其大小等于各维度的长度乘积
print("数组中包含的元素个数", arr.size)

# 数组维度大小，其大小等于ndarray.shape所包含元素的个数
print("数组维度大小", arr.ndim)

# 改变数组类型和形状
arr = arr.astype(np.str)
print(arr)

arr = arr.reshape([4, 5, 3])
print("改变形状", arr)

# 基本运算
a = a + 1
print(a)
a = a - 1
print(a)
a = 2 * a
print(a)
a = a / 2
print(a)

# 数组间的基本运算,是对应位置相加减，相乘除

# 使用循环生成数组
num_list = [k for k in range(0, 20, 3)]
print(num_list)

# 数组统计方法
"""
arr.mean: 计算平均数
arr.std和var: 计算标准差和方差
arr.sum: 对数组中全部或某轴方向进行求和
arr.max或min: 计算最大值最小值
arr.argmin和argmax: 最大和最小元素索引
arr.cumsum: 计算所有元素累加
arr.cumprod：计算所有元素乘积
属性axis: 指定维度求值

随机数组
np.random.seed(num)

均匀分布
np.random.rand(行，列)

正态分布
np.random.randn(行列)

随机打乱数组
np.random.shuffle(arr)  //只有行顺序会被打乱

随机选取元素
np.random.choice(arr, size)     //size选几个

线性代数
    dot矩阵乘法
    trace计算对角线元素和
    det计算行列式
    eig计算特征值特征向量
    inv计算方阵的逆


文件读写
    np.fromfile('file')
    np.load('path/file_name')
    

    

"""
