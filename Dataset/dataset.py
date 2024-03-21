import numpy as np
from torch.utils.data.dataset import Dataset
import copy


"""
def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1
"""
# 定义一个函数，根据数据集和类别数量进行分类
def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    # 创建一个空列表list 1，长度为num_classes类别数量
    for idx, (image, label) in enumerate(dataset):
    # 遍历数据集中的所有样本
        list1[label].append(idx)
        # 相同label的idx放在一起
    return list1
"""
    举个例子：
    假设有一个数据集包含了5个样本，它们的标签分别是0、1、1、2、0，即 [(image0, 0), (image1, 1), (image2, 1), (image3, 2), (image4, 0)]。
    如果我们调用 classify_label(dataset, 3)，表示有3个类别，那么函数将返回一个列表 list1，其中包含了3个子列表，分别存储了属于类别0、1和2的样本的索引。
    例如，list1 可能是 [[0, 4], [1, 2], [3]]，表示类别0的样本的索引为0和4，类别1的样本的索引为1和2，类别2的样本的索引为3。
                        0类      1类    2类
    也就是把所有的样本，每一类的索引，放在了同一个列表下，外边的列表是类列表，顺序是0、1.....。每一个小列表包含的都是相同label的idx
    把样本按类别分类了
"""

def show_clients_data_distribution(dataset, clients_indices: list, num_classes):
    # 定义一个函数，展示每个客户端的数据分布情况
    dict_per_client = []
    for client, indices in enumerate(clients_indices):
    # 创建一个长度为类别数量的列表，用于存储每个类别的样本数量
        nums_data = [0 for _ in range(num_classes)]
        # 遍历客户端的样本索引
        for idx in indices:
            label = dataset[idx][1]    # 获取样本的标签
            nums_data[label] += 1   # 对应类别的样本数量加1
        dict_per_client.append(nums_data)   # 将每个客户的样本分布添加到列表中
        print(f'{client}: {nums_data}') # 打印每个客户端的数据分布情况
    return dict_per_client
# 学一下


"""

def partition_train_teach(list_label2indices: list, ipc, seed=None):
# 定义一个函数，根据每个类别的样本索引列表和比例样本划分为训练集和教师集
    random_state = np.random.RandomState(0)
# 创建一个随机数生成器
    list_label2indices_teach = []
# 创建一个空表格 用来存储教师集

    # 遍历每个类别的样本索引列表
    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_teach.append(indices[:ipc])

    return list_label2indices_teach


def partition_unlabel(list_label2indices: list, num_data_train: int):
    random_state = np.random.RandomState(0)
    list_label2indices_unlabel = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_unlabel.append(indices[:num_data_train // 100])
    return list_label2indices_unlabel

"""


# FedIC
def partition_train_teach(list_label2indices: list, num_data_train: int, seed=None):
    random_state = np.random.RandomState(seed)
    list_label2indices_train = []
    list_label2indices_teach = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_train.append(indices[:num_data_train // 10])
        list_label2indices_teach.append(indices[num_data_train // 10:])
    return list_label2indices_train, list_label2indices_teach


def label_indices2indices(list_label2indices):
    indices_res = []
# 定义一个空列表，用来存储样本索引
    for indices in list_label2indices:  # 遍历标签索引列表
        indices_res.extend(indices)  # 将当前标签索引列表的所有元素添加到合并的列表上

    return indices_res
"""
    举个例子：
    list_label2indices = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    result = label_indices2indices(list_label2indices)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
"""


class Indices2Dataset(Dataset):  # 定义一个类，将样本索引转化为数据集对象
    def __init__(self, dataset):
        self.dataset = dataset  # 初始化样本集
        self.indices = None   # 初始化样本索引

    def load(self, indices: list):  # 加载指定的样本索引表
        self.indices = indices
    # 获取指定索引位置的样本
    def __getitem__(self, idx):
        idx = self.indices[idx]  # 获取实际的样本索引
        image, label = self.dataset[idx]     # 获取对应索引位置的样本和标签
        return image, label
    # 获取数据集的长度（样本数量）
    def __len__(self):
        return len(self.indices)   # 返回样本索引列表的长度


class TensorDataset(Dataset):  # 定义一个类表示带标签的数据集
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()  # 将输入图像的数据转化为浮点型，并去除梯度信息
        self.labels = labels.detach()  # 将输入标签数据去除梯度信息
    # 获得指定索引位置的样本和标签
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    # 获取数据集的长度，样本的数量
    def __len__(self):
        return self.images.shape[0]


def get_class_num(class_list):  # 定义一个函数，获取非零类别及其对应的样本数量
    index = []
    compose = []
    # 遍历类别列表
    for class_index, j in enumerate(class_list):
        if j != 0:  # 如果类别样本数量不为零
            index.append(class_index)   # 将类别索引添加到列表中去
            compose.append(j)  # 将样本数量添加到列表中
    return index, compose
