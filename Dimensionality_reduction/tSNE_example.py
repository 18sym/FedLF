# 导入必要的库：matplotlib用于绘图，sklearn.datasets用于加载数据集，sklearn.manifold.TSNE用于执行t-SNE降维
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE


# 通过sklearn的datasets模块加载手写数字数据集
digits = datasets.load_digits()

# 利用t-SNE算法将高维的手写数字数据降维到2维，random_state=33用于确保实验可重复
# n_components=2表示目标是降维到二维空间
X_tsne = TSNE(n_components=2, random_state=33).fit_transform(digits.data)

# 创建一个图像窗口，设置大小为10x6
plt.figure(figsize=(10, 6))

# 遍历降维后的数据，为每一个数据点在图形上生成一个文本标签
# X_tsne[i, 0]和X_tsne[i, 1]代表每个点在二维空间的横纵坐标
# str(digits.target[i])是该数据点的标签（即数字0-9），用文本的形式表示
# color=plt.cm.Set1(digits.target[i] / 10.)设置文本的颜色，不同的数字会有不同的颜色
# fontdict设置文本的字体为粗体和大小为9
for i in range(len(X_tsne)):
    plt.text(X_tsne[i, 0], X_tsne[i, 1], str(digits.target[i]),
             color=plt.cm.Set1(digits.target[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})

# 在for循环之后，但在plt.show()之前，添加以下两行代码来手动设置显示范围
plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max())
plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max())
# 隐藏x轴和y轴的刻度，因为我们关心的是数据点的分布而不是具体坐标
# plt.xticks([])
# plt.yticks([])

# 设置图表标题
plt.title('t-SNE visualization of digits data')

# 显示图表
plt.show()