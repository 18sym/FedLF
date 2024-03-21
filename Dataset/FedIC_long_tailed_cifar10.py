import numpy as np  # 导入numpy库，用于数值计算
from Dataset.dataset import label_indices2indices   # label_indices2indices 所有样本的样本索引列表
import copy  # 导入copy模块，用于深拷贝列表


def _get_img_num_per_cls(list_label2indices_train, num_classes, imb_factor, imb_type):
    # 定义一个函数，用于计算每个类别的样本数量
    img_max = len(list_label2indices_train) / num_classes  # 计算每个类别的最大样本数量
    img_num_per_cls = []   # 创建一个空列表，用于存储每个类别的样本数量
    if imb_type == 'exp':  # 根据不同的imb_type计算每个类别的样本数量
        for _classes_idx in range(num_classes):
            # 使用指数函数计算样本数量
            num = img_max * (imb_factor**(_classes_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))  # 将计算得到的样本数量添加到列表中
    return img_num_per_cls


# 定义函数，用来生成长尾数据分布
def train_long_tail(list_label2indices_train, num_classes, imb_factor, imb_type):
    # 调用label_indices2indices函数，将标签索引转换为样本索引，所有的样本索引
    new_list_label2indices_train = label_indices2indices(copy.deepcopy(list_label2indices_train))
    # new_list_label2indices_train 为所有的样本索引
    # 调用_get_img_num_per_cls函数，计算每个类别的样本数量
    img_num_list = _get_img_num_per_cls(copy.deepcopy(new_list_label2indices_train), num_classes, imb_factor, imb_type)
    print('img_num_class')  # 打印提示信息
    # img_num_class
    # [5000, 3237, 2096, 1357, 878, 568, 368, 238, 154, 100]
    print(img_num_list)  # 打印每个类别的样本数量列表

    list_clients_indices = []  # 空列表，存储客户端的样本索引列表
    classes = list(range(num_classes))  # 生成类别列表
    for _class, _img_num in zip(classes, img_num_list):
        # for in zip 并行遍历，同时遍历两个列表
        indices = list_label2indices_train[_class]
        # 获取当前类别的样本索引列表
        np.random.shuffle(indices)
        #  对样本索引列表所及打乱
        idx = indices[:_img_num]  # 根据每个类别的样本数量截取样本索引
        list_clients_indices.append(idx)  # 将生成的样本索引添加到到客户端的样本索引列表中
    num_list_clients_indices = label_indices2indices(list_clients_indices)
    print('All num_data_train') # 打印提示信息
    print(len(num_list_clients_indices))  # 打印总的样本数量
    return list_clients_indices  # 返回每个类别的样本数量列表和客户端的样本索引列表
# img_num_list,




