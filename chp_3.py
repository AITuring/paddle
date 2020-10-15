###################   线性回归从零开始   #####################
import paddle
from matplotlib import pyplot as plt
import numpy as np
import random
from d2lzh_paddle import *

# 使用动态图
paddle.disable_static()

# 3.2.1 生成数据集
# y = Xw + b + eps
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = paddle.rand(shape=[num_examples, num_inputs], dtype='float32')
y_labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
y_labels += paddle.to_tensor(np.random.normal(0, 0.01, size=y_labels.size()),
                             dtype='float32')

print(features[0], y_labels[0])

set_figsize()
plt.scatter(features[:, 1].numpy(), y_labels.numpy(), 1)
# plt.show()

# 3.2.2 读取数据
batch_size = 10
for X, y in data_iter(batch_size, features, y_labels):
    print(X, y)
    break

# 3.2.3 初始化模型参数
w = paddle.to_tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype='float32', stop_gradient=False)
b = paddle.to_tensor(paddle.zeros(shape=[1], dtype='float32'), stop_gradient=False)
print(w, b)

# 3.2.7 训练模型
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):   # 训练模型共num_epochs个迭代周期
    # 在每次迭代中，训练所有样本一次
    for X, y in data_iter(batch_size, features, y_labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        # 梯度清零
        w.clear_gradient()
        b.clear_gradient()
    train_l = loss(net(features, w, b), y_labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().numpy()))



















