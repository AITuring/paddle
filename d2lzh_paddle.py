##############   utils   ################
import matplotlib
import matplotlib.pyplot as plt
import random
import sys
import paddle
from paddle import nn

def set_figsize(figsize=(3.5, 2.5)):
    matplotlib.rcParams['backend'] = 'SVG'
    plt.rcParams['figure.figsize'] = figsize

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)   # 打乱顺序
    for i in range(0, num_examples, batch_size):
        j = paddle.to_tensor(indices[i: min(i + batch_size, num_examples)], dtype='int64')
        yield paddle.index_select(features, axis=0, index=j), paddle.index_select(labels, axis=0, index=j)

# 3.2.4 定义模型
def linreg(X, w, b):
    return paddle.mm(X, w) + b

# 3.2.5 定义损失函数
def squared_loss(y_hat, y):
    y = paddle.reshape(y, shape=y_hat.size())
    return (y_hat - y) ** 2 / 2

# 3.2.6 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param_data = param.numpy()
        param_data -= lr * param.grad / batch_size
        param.set_value(param_data)
# 3.5.2 读取小批量数据
def load_data_mnist(batch_size):
    mnist_train = paddle.vision.datasets.MNIST(mode='train')
    mnist_test = paddle.vision.datasets.MNIST(mode='test')
    if sys.platform.startswith('win'):
        num_workers = 0  # 不用额外进程来加速读取数据
    else:
        num_workers = 4
    train_data = paddle.io.DataLoader(
        mnist_train,
        return_list=True,
        places=paddle.CPUPlace(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_data = paddle.io.DataLoader(
        mnist_test,
        return_list=True,
        places=paddle.CPUPlace(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return train_data,test_data

# 5.1.1 二维卷积互相关运算
def corr2d(X, K):
    h, w = K.shape
    Y = paddle.zeros(shape=[X.shape[0] - h + 1, X.shape[1] - w + 1], dtype='float32')
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = paddle.sum(X[i: i + h, j: j + w] * K)
    return Y


