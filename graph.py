# 导入相关包
import paddle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 版本检测
print(paddle.__version__)


# minist加载数据
# mode来标识是训练数据还是测试数据集
mnist_train = paddle.vision.datasets.MNIST(mode='train')
mnist_test =  paddle.vision.datasets.MNIST(mode='test')

# 查看数据集大小
# 训练集中和测试集中的每个类别的图像数分别为6,000和1,000。因为有10个类别，所以训练集和测试集的样本数分别为60,000和10,000
print(len(mnist_train)) # 60000
print(len(mnist_test)) # 10000
print(type(mnist_train)) # <class 'paddle.vision.datasets.mnist.MNIST'>

# 取训练集数据查看，通过下标访问任意一个样本
train_data0, train_label_0 = mnist_train[0][0],mnist_test[0][1]
print(train_data0.shape,train_data0.dtype) #(1, 28, 28) float32
print(train_label_0.shape,train_label_0.dtype) # (1,) int64
train_data0 = train_data0.reshape([28,28])
# 绘图查看
plt.figure(figsize=(2,2))
plt.imshow(train_data0, cmap=plt.cm.binary)
print('train_data0 label is: ' + str(train_label_0)) # train_data0 label is: [7]

# TODO 整体图片展示做不做，回家用win10做吧

# TODO 批量读取数据需要进一步调试

# test
# 小批量读取数据
batch_size = 256

# train_loader = paddle.io.DataLoader(mnist_train, places=paddle.CPUPlace(), batch_size=64, shuffle=True)
# # 开启动态图
# paddle.disable_static()
# # 模型搭建
# # 针对顺序的线性网络结构我们可以直接使用Sequential来快速完成组网，可以减少类的定义等代码编写
# mnist = paddle.nn.Sequential(
#     # paddle.nn.Flatten(),
#     paddle.nn.Linear(784,512),
#     paddle.nn.ReLU(),
#     paddle.nn.Dropout(0.2),
#     paddle.nn.Linear(512,10)
# )
#
#
# # 模型训练
#
# # 封装模型
# # 预计模型结构生成模型实例，便于后续配置、训练、验证
# model = paddle.Model(mnist)
# # 模型训练相关配置，准备损失计算方法，优化器和精度计算方法
# # 模型可视化
# # model.summary((1,28,28))
# # paddle.summary(mnist,(1,28,28))
# model.prepare(
#     paddle.optimizer.Adam(parameters=mnist.parameters()),
#     paddle.nn.CrossEntropyLoss(),
#     paddle.metric.Accuracy()
# )
# # 开始模型训练
# # 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
# model.fit(
#     mnist_train,
#     epochs=10,
#     batch_size=32,
#     verbose=1
# )
# # 模型评估
# model.evaluate(mnist_test,verbose=0)
# print(model.evaluate(mnist_test,verbose=0))
#
# # 模型预测
# predict = model.predict(mnist_test)