import paddle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import paddle.nn.functional as F


# 版本检测
paddle.disable_static()
print(paddle.__version__)

# minist加载数据
# mode来标识是训练数据还是测试数据集
mnist_train = paddle.vision.datasets.MNIST(mode='train', chw_format=False)
mnist_test =  paddle.vision.datasets.MNIST(mode='test', chw_format=False)

# train_reader = paddle.batch(mnist_train, batch_size=8)
#
# for batch_id, data in enumerate(train_reader()):
#     # 获得图像数据，并转为float32类型的数组
#     img_data = np.array([x[0] for x in data]).astype('float32')
#     # 获得图像标签数据，并转为float32类型的数组
#     label_data = np.array([x[1] for x in data]).astype('float32')
#     # 打印数据形状
#     print("图像数据形状和对应数据为:", img_data.shape, img_data[0])
#     print("图像标签形状和对应数据为:", label_data.shape, label_data[0])
#     break
#
# print("\n打印第一个batch的第一个图像，对应标签数字为{}".format(label_data[0]))
# # 显示第一batch的第一个图像
# import matplotlib.pyplot as plt
# img = np.array(img_data[0]+1)*127.5
# img = np.reshape(img, [28, 28]).astype(np.uint8)
#
# plt.figure("Image") # 图像窗口名称
# plt.imshow(img)
# plt.axis('on') # 关掉坐标轴为 off
# plt.title('image') # 图像题目
# plt.show()

print(len(mnist_train))
print(len(mnist_test))

# 获取第一个样本图形和标签
feature,label = mnist_train[0]
print(feature.shape,feature.dtype)

# 模型搭建
# 针对顺序的线性网络结构我们可以直接使用Sequential来快速完成组网，可以减少类的定义等代码编写
mnist = paddle.nn.Sequential(
    paddle.nn.Linear(784,512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512,10)
)


# 模型训练
# 开启动态图
paddle.disable_static()
# 预计模型结构生成模型实例，便于后续配置、训练、验证
model = paddle.Model(mnist)
# 模型训练相关配置，准备损失计算方法，优化器和精度计算方法
# 模型可视化
model.summary((1,28,28))
model.prepare(
    paddle.optimizer.Adam(parameters=mnist.parameters()),
    paddle.nn.CrossEntropyLoss(),
    paddle.metric.Accuracy()
)
# 开始模型训练
model.fit(
    mnist_train,
    epochs=5,
    batch_size=32,
    verbose=1
)
# # 模型评估
# model.evaluate(mnist_test,verbose=0)
# print(model.evaluate(mnist_test,verbose=0))