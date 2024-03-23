import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from options import args_parser
from Model.Resnet8 import *
# 相关库的导入和 ResNet_cifar 的定义应该已经完成。

# 加载 CIFAR-10 数据集
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

# 实例化模型，此处 args 对应代码缺少相关定义，需要你补充或使用默认参数
args = args_parser()  # 需要自定义函数 args_parser() 或直接使用实参
model = ResNet_cifar(resnet_size=8, scaling=4, save_activations=False,
                     group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False,
                     num_classes=args.num_classes)

# 在你提供的类中，默认没有 load_state_dict, 如果你有预训练的模型权重，可以在这里加载
# model.load_state_dict(torch.load('model_weights.pth'))

model.eval()  # 设置模型为评估模式

# GPU支持
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 提取特征
features = []
labels = []
with torch.no_grad():
    for data in trainloader:
        images, label_batch = data
        images = images.to(device)
        feature_batch, _ = model(images)
        features.append(feature_batch.cpu().numpy())
        labels.append(label_batch.numpy())

# 将所有分批处理的特征和标签合并在一起
features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

# 你已经有了通过 ResNet 提取的特征，现在进行 t-SNE
n_samples = 1000  # 由于计算成本，通常我们仅仅使用部分样本进行 t-SNE
X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features[:n_samples])
y = labels[:n_samples]

# 可视化使用 t-SNE 转换后的特征
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(scatter)
plt.title('t-SNE visualization of CIFAR-10 features extracted by ResNet')
plt.show()