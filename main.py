"""
Tensorflow结构：

    Tensorflow中主要是由构建图得阶段和执行图的阶段两部分组成
    （图）在构建阶段：数据与操作的执行步骤被描述成为图（流程图）
    （会话）在执行阶段：使用绘画去执行构建好的图
    张量：是Tensorflow中的基本数据对象
    节点：提供图当中执行的操作

    图结构：图包含了一组tf.Operation代表的计算单元对象和tf.Tensor代表的计算单元之间流动的数据

    图的相关操作：
        1. 默认图
        通过调用tf.get_default_graph()访问，将操作添加到默认图中，直接创建OP即可
        op,sess都含有graph属性，默认在一张图中
        2. 自定义图


"""

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def add_demo():

    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c = a_t + b_t
    with tf.compat.v1.Session() as sess:
        c_value = sess.run(c)
        print("c_value:", c_value)
    print(a_t)
    print(b_t)
    print(c)

    # 查看默认图的方法：
    # 1.调用方法

    # 2. 查看属性


if __name__ == "__main__":
    add_demo()
