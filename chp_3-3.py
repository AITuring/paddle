import paddle
import numpy as np
import paddle.fluid as fluid
import paddle.nn as nn
from paddle.nn import initializer
import paddle.optimizer as optim

paddle.disable_static()

# 3.3.1 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = paddle.rand(shape=[num_examples, num_inputs], dtype='float32')
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += paddle.to_tensor(np.random.normal(0, 0.01, size=labels.size()), dtype='float32')

# print(features, '\n', labels)

# 3.3.2 读取数据
batch_size = 10
data_set = fluid.io.TensorDataset([features, labels])
data_iter = fluid.io.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True, places=fluid.CUDAPlace(0))

for X, y in data_iter:
    # print(X, y)
    break

# 3.3.3 定义模型
class LinearNet(nn.Layer):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net)

# 使用Sequential搭建网络
# Method 1:
# net = nn.Sequential(
#     nn.Linear(num_inputs, 1)
# )

# Mothod 2:
net = nn.Sequential()
net.add_sublayer('linear', nn.Linear(num_inputs, 1))

# Method 3:
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#     ('linear', nn.Linear(num_inputs, 1))
# ]))

print(net)

for param in net.parameters():
    print(param)

# 3.3.4 初始化模型参数
# 设置全局参数初始化
fluid.set_global_initializer(initializer.Uniform(), initializer.Constant())

# 3.3.5 定义损失函数
loss = nn.MSELoss()

# 3.3.6 定义优化算法
optimizer = optim.SGD(learning_rate=0.03, parameters=net.parameters())
print(optimizer)

# 设置不同自网络的学习率（待修改）
# optimizer = optim.SGD([
#     {'params': net._sub_layers1.paramaters()},
#     {'params': net._sub_layers2.paramaters(), 'lr': 0.01}
# ], learning_rate=0.03)

# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1

# 3.3.7 训练模型
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, paddle.reshape(y, shape=[-1, 1]))
        optimizer.clear_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.numpy()))

dense = net
print(true_w, dense.linear.weight.numpy())
print(true_b, dense.linear.bias.numpy())




















