from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

import Network

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor()
)

# 数据集大小
# print(len(train_data))  # 60000
# print(len(test_data))   # 10000

# train_data[i]: (image, label)
# image: torch.Tensor, shape = [1, 28, 28]
# label: int, range = 0~9
# 0 <= i < 60000

# image, t = train_data[0]
# image = image.reshape(1, 784)
# label = np.zeros((1, 10))
# label[0, t] = 1

# g1 = network.gradient(image, label)
# g2 = network.gradient2(image, label)

# print("g1['B1']")
# print(g1['B1'])
# print("g2['B1']")
# print(g2['B1'])


# 随机选择batch_size个图像，计算并记录每个图像下loss对各项训练参数的梯度
# 求平均后更新参数，新参数 -= learning_rate*gradient
rounds = 10000
batch_size = 600

network = Network.network(784, 50, 10)

train_loss = []
train_accuracy = []

for j in range(rounds):
    grad = {
        'W1': np.zeros_like(network.params['W1']),
        'W2': np.zeros_like(network.params['W2']),
        'B1': np.zeros_like(network.params['B1']),
        'B2': np.zeros_like(network.params['B2'])
    }

    temp_loss = 0
    temp_accutacy = 0

    indices = np.random.choice(len(train_data), size=batch_size, replace=False)
    for i in indices:
        image, t = train_data[i]
        image = image.reshape(1, 784)
        label = np.zeros((1, 10))
        label[0, t] = 1

        temp_grad = network.gradient(image, label)
        for k in grad:
            grad[k] += temp_grad[k] / batch_size

        temp_accutacy += network.accuracy(image, t) / batch_size
        temp_loss += network.loss(image, t) / batch_size

    if j % 100 == 0:
        print(j)
    train_loss.append(temp_loss)
    train_accuracy.append(temp_accutacy)

    network.update(grad)

for i in range(rounds):
    print("i: " + str(i))
    print("train_loss[i]: " + str(train_loss[i]))
    print("train_accuracy[i]: " + str(train_accuracy[i]))

###############################################################
epochs = np.arange(len(train_loss))   # 0..99，也就是 batch 编号

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# loss 曲线
axes[0].plot(epochs, train_loss, color='tab:red')
axes[0].set_title("Train Loss")
axes[0].set_xlabel("Batch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

# accuracy 曲线
axes[1].plot(epochs, train_accuracy, color='tab:blue')
axes[1].set_title("Train Accuracy")
axes[1].set_xlabel("Batch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].grid(True)

plt.tight_layout()
plt.show()                                   # 弹窗显示
###############################################################

np.savez("params.npz", **network.params)