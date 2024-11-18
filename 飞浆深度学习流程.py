# 深度学习构建模型和完成训练的过程
"""
    数据处理(获取数据完成预处理操作（数据校验，格式转化等等）)，保证模型可读取

    模型设计（网络结构设计，相当于模型的假设空间，即模型能够表达的关系集合）

    训练配置（设定模型采用得分寻解算法，并指定计算资源）
        ①生成模型实例
        ②设置成训练模式（还有预测模式）
        ③读取训练和测试数据集
        ④优化算法 = SGD，学习率 = 0.01
    训练过程（每轮都包括向前计算，损失函数，向后传播三步）
        ①准备数据
        ②向前计算（调用模型实例，运行forward函数）
        ③计算损失
        ④计算梯度
        ⑤更新参数

    模型保存，将训练好的模型进行保存，模型预测时调用



飞浆中的动态图与静态图
    静态图模式：先编译后执行，预先定义完整的网络结构，在对其进行编译优化后再执行计算结果
    动态图模式：解析式执行方式，无需预先定义完整的网络结构

"""

