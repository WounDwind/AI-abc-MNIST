import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
import Network


def load_network(param_path="params.npz"):
    """按指定方式载入网络结构并加载已训练好的参数"""
    network = Network.network(784, 50, 10)
    data = np.load(param_path)
    network.params['W1'] = data['W1']
    network.params['W2'] = data['W2']
    network.params['B1'] = data['B1']
    network.params['B2'] = data['B2']
    return network


def test(network, test_data, test_size=200, seed=0):
    """
    从 test_data 中随机选 test_size 个样本，
    独立计算平均 loss 与 accuracy（不依赖 Network 里那个有 bug 的 accuracy）。
    返回 (avg_loss, accuracy, wrong_list)
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(test_data), size=test_size, replace=False)

    total_loss = 0.0
    correct = 0
    wrong = []

    for idx in indices:
        image, t = test_data[idx]                 # image: [1,28,28] tensor, t: int
        x = image.reshape(1, 784).numpy()         # (1, 784)

        prob = network.predict(x)[0]              # (10,)
        loss = -np.log(prob[t] + 1e-7)
        total_loss += loss

        pred = int(np.argmax(prob))
        if pred == t:
            correct += 1
        else:
            wrong.append({
                'img':   image.numpy().squeeze(),  # (28,28)
                'true':  int(t),
                'pred':  pred,
                'p_pred': float(prob[pred]),
                'p_true': float(prob[t]),
            })

    avg_loss = total_loss / test_size
    acc      = correct / test_size

    print(f"test_size = {test_size}")
    print(f"avg loss  = {avg_loss:.4f}")
    print(f"accuracy  = {acc:.4f}  ({correct}/{test_size})")
    print(f"wrong     = {len(wrong)}")

    return avg_loss, acc, wrong


def plot_wrong(wrong, test_size, acc,
               save_path="wrong_predictions.png", cols=8):
    """画出所有被错误预测的样本：标题显示预测标签 P、真实标签 T 及概率"""
    n = len(wrong)
    if n == 0:
        print("No wrong predictions.")
        return

    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 1.4, rows * 1.8))
    axes = np.array(axes).reshape(-1)

    for ax, item in zip(axes, wrong):
        ax.imshow(item['img'], cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(
            f"T:{item['true']}  P:{item['pred']}\n"
            f"p_true={item['p_true']:.2f}  p_pred={item['p_pred']:.2f}",
            fontsize=7, color='red'
        )
        ax.axis('off')

    # 关闭多余的子图
    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle(
        f"Wrong predictions: {n}/{test_size}   accuracy={acc:.4f}",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    # 1) 载入参数
    network = load_network("params.npz")

    # 2) 载入 MNIST 测试集
    test_data = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    # 3) 随机选 test_size 个样本测试
    test_size = 10000
    avg_loss, acc, wrong = test(network, test_data,
                                test_size=test_size, seed=42)

    # 4) 画出所有错误样本
    plot_wrong(wrong, test_size, acc)