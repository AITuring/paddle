import paddle
import sys
import time
import d2lzh_paddle as d2l
import numpy as np

# 读取数据
batch_size = 256
train_data, test_data = d2l.load_data_mnist(batch_size)
print(train_data)
# 初始化模型参数
num_inputs = 784
num_outputs = 10

# TODO 现在导入数据可能因为数据结构不同对不上了，也不想做了，回去再研究研究，paddle.to_tensor
# W = paddle.random.normal(scale=0.01, shape = (num_inputs, num_outputs))
b = paddle.zeros(num_outputs)

# print('W',W)
print('b',b)
