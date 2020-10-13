import paddle
import paddle.nn.functional as F
import numpy as np

# 版本检测
paddle.disable_static()
print(paddle.__version__)

# minist加载数据
mnist_train = paddle.vision.datasets.MNIST(mode='train', chw_format=False)
mnist_test =  paddle.vision.datasets.MNIST(mode='test', chw_format=False)

print(len(mnist_train))
print(len(mnist_test))

# 获取第一个样本图形和标签
feature,label = mnist_train[0]
print(feature.shape,feature.dtype)

# # 模型搭建
# mnist = paddle.nn.Sequential(
#     paddle.nn.Linear(784,512),
#     paddle.nn.ReLU(),
#     paddle.nn.Dropout(0.2),
#     paddle.nn.Linear(512,10)
# )
#
# # 模型训练
# # 开启动态图
# paddle.disable_static()
# # 预计模型结构生成模型实例，便于后续配置、训练、验证
# model = paddle.Model(mnist)
# # 模型训练相关配置，准备损失计算方法，优化器和精度计算方法
# model.prepare(
#     paddle.optimizer.Adam(parameters=mnist.parameters()),
#     paddle.nn.CrossEntropyLoss(),
#     paddle.metric.Accuracy()
# )
# # 开始模型训练
# model.fit(
#     mnist_train,
#     epochs=5,
#     batch_size=32,
#     verbose=1
# )
# # 模型评估
# model.evaluate(mnist_test,verbose=0)
# print(model.evaluate(mnist_test,verbose=0))