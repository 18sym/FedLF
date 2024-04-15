from torchvision import datasets
# datasets 模块用于，加载和使用预定义的数据集

from torchvision.transforms import ToTensor, transforms
# ToTensor模块用于将PIL图片或者numpy数组转化为Tensor
# transforms模块提供了一系列的图像比那换操作，可用于数据增强

from options import args_parser
# argus_parser，用于解析命令行参数

from Dataset.long_tailed_cifar10 import train_long_tail
# 自己定义的函数，可用于处理长尾数据集CIFAR-10

from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset, TensorDataset, \
    get_class_num
# 从自己定义的包中导入定义文件的函数

from Dataset.sample_dirichlet import clients_indices
# 导入数据采样方法

from Dataset.Gradient_matching_loss import match_loss
# 用来计算梯度损失

import numpy as np
# 导入numpy，用于执行各种计算的库

from torch import stack, max, eq, no_grad, tensor, unsqueeze, split
# 从 torch包导入多个函数和上下文管理器，用于张量操作、比较、禁用梯度计算率

from torch.optim import SGD
# 导入SGD实现随机梯度下降优化算法

from torch.nn import CrossEntropyLoss
# 用于计算交叉熵损失，长用于多分类问题

from torch.utils.data.dataloader import DataLoader
# DataLoader用于高效加载数据

from Model.Resnet8 import ResNet_cifar
# 可能是为CIFAR数据集定制的ResNet8 模型

from tqdm import tqdm
# 从tqdm 导入 tqdm，用于循环显示进度

import copy
# 导入copy模块，用于执行对象的浅复制和深复制

import torch
# 导入torch库，pytorch的核心库，用于张量计算和自动微分

import random
# 用于生成随机数

import torch.nn as nn
# 导入random模块，并重命名为nn，提供了构建神经网络的类和函数

import time
# 导入时间模块，用于测量时间

from Dataset.param_aug import DiffAugment
# 从Dataset包下的param_aug模块导入DiffAugment类，可能是一个实现差异化数据增强的类。
from algorithm.FedIC import disalign
from algorithm.FedBN import FedBN



class Global(object):
    def __init__(self,
                 num_classes: int,  # 数据集中的类别总数
                 device: str,  # 计算设备类型，如’gpu‘或者’cpu‘
                 args,  # 命令行参数对象， 用于获取训练配置
                 num_of_feature):  # 特征的数量
        self.device = device  # 存储计算设备类型
        self.num_classes = num_classes  # 存储数据中的类别总数
        self.fedavg_acc = []  # 联邦平均精度列表
        self.fedavg_many = []  # 联邦平均在"多数"类别上的精度列表
        self.fedavg_medium = []  # 联邦平均在"中等"类别上的精度列表
        self.fedavg_few = []  # 联邦平均在“少数”类别上的精度列表
        self.ft_acc = []  # 微调之后的精度列表
        self.ft_many = []  # 微调后在“多类”类别上的精度列表
        self.ft_medium = []  # “中类”
        self.ft_few = []  # ‘少类’
        self.num_of_feature = num_of_feature  # 存储每个类别的特征数量

        # 随机初始化合成特征张量，形状为(类别数*每类特征数, 256)，需要梯度，用于优化
        # torch.randn(size , dtype , requires_grad=Ture, device=args.device)
        self.feature_syn = torch.randn(size=(args.num_classes * self.num_of_feature, 256), dtype=torch.float,
                                       requires_grad=True, device=args.device)

        # 创建合成标签，每个类别有num_of_feature个相同的标签，不需要梯度
        self.label_syn = torch.tensor([np.ones(self.num_of_feature) * i for i in range(args.num_classes)],
                                      dtype=torch.long,
                                      requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
        """
        创建一个PyTorch张量label_syn，该张量包含合成数据的标签。让我们逐步解析这个表达式：
        1: np.ones(self.num_of_feature) * i: 对于每个类别i（从0到args.num_classes - 1），这个表达式创建了一个长度为self.num_of_feature的数组，数组中的每个元素都是类别i。这意味着，如果self.num_of_feature是3，并且有10个类别（即args.num_classes为10），那么当i为0时，这个表达式生成的数组将是[0, 0, 0]；当i为1时，生成的数组将是[1, 1, 1]，以此类推。
        2: [np.ones(self.num_of_feature) * i for i in range(args.num_classes)]: 这是一个列表推导式，它遍历从0到args.num_classes - 1的所有整数i，对于每个i，使用上一步中的表达式生成一个数组，并将这些数组收集到一个列表中。继续上面的例子，最终生成的列表将是[[0, 0, 0], [1, 1, 1], ..., [9, 9, 9]]。
        3: torch.tensor(..., dtype=torch.long, requires_grad=False, device=args.device): 这部分代码将上一步生成的列表转换为一个PyTorch张量。dtype=torch.long指定了张量的数据类型为长整型（适用于标签）。requires_grad=False表示在反向传播时，不需要计算这个张量的梯度（因为标签是固定的，不需要优化）。device=args.device指定了张量应该存储在哪个设备上，例如CPU或GPU。
        4: .view(-1): 最后，.view(-1)方法用于将张量展平成一维。在上面的例子中，原始的二维列表（或者说，转换成张量后的二维结构）是[[0, 0, 0], [1, 1, 1], ..., [9, 9, 9]]，展平后变成[0, 0, 0, 1, 1, 1, ..., 9, 9, 9]。-1在这里是一个特殊值，告诉PyTorch自动计算这个维度的大小以保持元素总数不变。
        总结来说，这行代码的目的是创建一个包含所有类别标签的一维张量，每个类别有self.num_of_feature个重复的标签。这个张量后续可能用于与合成特征一起，作为模型训练的目标标签。
        """

        # 为合成特征数据创建SGD优化器，学习率从命令行参数获取
        # SGD(需要优化的参数（张量），lr， momentum（动量）， dampening（阻尼值，防止动量引起的震荡）， nesterov（默认为false，采用动量更新规则）)
        self.optimizer_feature = SGD([self.feature_syn, ], lr=args.lr_feature)  # optimizer_img for synthetic data

        # 创建交叉熵损失函数，并将其移动到指定的设备上
        self.criterion = CrossEntropyLoss().to(args.device)

        # 初始化自定义的ResNet_cifar模型，并将其移动到指定的设备上
        self.syn_model = ResNet_cifar(resnet_size=8, scaling=4,
                                      save_activations=False, group_norm_num_groups=None,
                                      freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(device)

        # 创建一个线性层（全连接层），用于从合成特征中进行分类，输出类别数为10，模型移动到指定设备
        self.feature_net = nn.Linear(256, 10).to(args.device)
        #

    def update_feature_syn(self, args, global_params, list_clients_gradient):
        # 获取网络参数
        feature_net_params = self.feature_net.state_dict()
        for name_param in reversed(global_params):
            # 如果参数名为’classifier.bias‘,则更新特征网络的偏执
            if name_param == 'classifier.bias':
                feature_net_params['bias'] = global_params[name_param]
            # 如果参数’classifier.weight‘，则更新特征网络的权重
            if name_param == 'classifier.weight':
                feature_net_params['weight'] = global_params[name_param]
                break
        # 加载特征网络参数
        self.feature_net.load_state_dict(feature_net_params)
        self.feature_net.train()

        # 获取特征网络参数
        net_global_parameters = list(self.feature_net.parameters())
        """
        初始化 真实全局梯度字典 从0开始
        # 创建一个字典，键为0、1、2，值为空列表
        gw_real_all = {class_index: [] for class_index in range(3)}
        print(gw_real_all)

        {0: [], 1: [], 2: []}
        """
        gw_real_all = {class_index: [] for class_index in range(self.num_classes)}

        # 遍历客户端梯度列表
        for gradient_one in list_clients_gradient:
            for class_num, gradient in gradient_one.items():
                gw_real_all[class_num].append(gradient)

        # 初始化真实平均梯度字典
        gw_real_avg = {class_index: [] for class_index in range(args.num_classes)}
        # aggregate the real feature gradients

        # 聚合真实的梯度
        for i in range(args.num_classes):
            gw_real_temp = []
            list_one_class_client_gradient = gw_real_all[i]

            if len(list_one_class_client_gradient) != 0:
                weight_temp = 1.0 / len(list_one_class_client_gradient)
                for name_param in range(2):
                    list_values_param = []
                    for one_gradient in list_one_class_client_gradient:
                        list_values_param.append(one_gradient[name_param] * weight_temp)
                    value_global_param = sum(list_values_param)
                    gw_real_temp.append(value_global_param)
                gw_real_avg[i] = gw_real_temp

    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        # 初始化fedavg全局参数
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        # 对于第一个客户端的本地模型参数中的每个参数，进行迭代
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            # 计算全局参数
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
                # 将当前参数在当前客户端的值乘以该客户端的数据量，并将结果添加到 list_values_param 列表中。
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            # 计算所有客户端中当前参数的平均值，这是通过将所有客户端中的参数值总和除以所有客户端的数据量总和来实现的。
            fedavg_global_params[name_param] = value_global_param
            # 将全局参数中当前参数的值设置为上一步计算的全局平均值
        return fedavg_global_params  # 返回计算得到的全局参数，这些参数已经是所有客户端模型参数的平均值。

    def global_eval_more(self, fedavg_params, data_test, batch_size_test, a):
        # 首先加载模型参数
        self.syn_model.load_state_dict(fedavg_params)
        self.syn_model.eval()

        # 定义类别的阈值
        majority_threshold = 1500
        minority_threshold = 200

        # 初始化精确度计算所需的变量
        num_corrects_majority, num_samples_majority = 0, 0
        num_corrects_medium, num_samples_medium = 0, 0
        num_corrects_minority, num_samples_minority = 0, 0

        img_num_class = a

        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test, shuffle=False)
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)

                # 此处假设模型返回结果的方式
                _, outputs = self.syn_model(images)
                _, predicts = max(outputs, -1)

                for label, predict in zip(labels, predicts):
                    # 根据label确定样本所属的类别范围
                    samples_num = img_num_class[label.item()]
                    correct = predict.cpu().item() == label.cpu().item()
                    if samples_num > majority_threshold:
                        num_samples_majority += 1
                        num_corrects_majority += correct
                    elif samples_num < minority_threshold:
                        num_samples_minority += 1
                        num_corrects_minority += correct
                    else:
                        num_samples_medium += 1
                        num_corrects_medium += correct

        # 计算并返回三个类别（多数、中等、少数）的精确度，保留小数点后四位
        accuracy_majority = round(num_corrects_majority / num_samples_majority, 4) if num_samples_majority > 0 else 0
        accuracy_medium = round(num_corrects_medium / num_samples_medium, 4) if num_samples_medium > 0 else 0
        accuracy_minority = round(num_corrects_minority / num_samples_minority, 4) if num_samples_minority > 0 else 0

        # 返回计算结果
        return accuracy_majority, accuracy_medium, accuracy_minority



    def global_eval(self, fedavg_params, data_test, batch_size_test):
        # 加载FedAVG参数到合成模型
        self.syn_model.load_state_dict(fedavg_params)
        self.syn_model.eval()
        with no_grad():
            # 构建测试数据加载器
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            # 进行全局评估
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.syn_model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        # 下载合成模型参数
        return self.syn_model.state_dict()


class Local(object):
    def __init__(self,
                 data_client,
                 class_list: int):
        # 初始化函数，接受数据集和类别列表作为参数
        args = args_parser()  # 解析命令行参数

        self.data_client = data_client  # 客户端数据集

        self.device = args.device  # 设备
        self.class_compose = class_list  # 类别列表

        self.criterion = CrossEntropyLoss().to(args.device)  # 交叉熵损失函数

        # 初始化本地模型，使用 ResNet 架构
        self.local_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(
            args.device)

        # 初始化优化器，使用 SGD
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)

    def compute_gradient(self, global_params, args):
        # compute C^k 计算梯度的函数
        # 获取类别和每个类别的样本数
        list_class, per_class_compose = get_class_num(self.class_compose)  # class组成
        # 收集所有图像和标签
        images_all = []
        labels_all = []
        indices_class = {class_index: [] for class_index in list_class}

        images_all = [unsqueeze(self.data_client[i][0], dim=0) for i in range(len(self.data_client))]
        labels_all = [self.data_client[i][1] for i in range(len(self.data_client))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        def get_images(c, n):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        # 加载全局参数到本地模型
        self.local_model.load_state_dict(global_params)

        self.local_model.eval()
        self.local_model.classifier.train()
        net_parameters = list(self.local_model.classifier.parameters())
        criterion = CrossEntropyLoss().to(args.device)

        # gradients of all classes
        # 所有类别的梯度
        truth_gradient_all = {index: [] for index in list_class}
        truth_gradient_avg = {index: [] for index in list_class}

        # choose to repeat 10 times 重复10次计算
        for num_compute in range(10):
            for c, num in zip(list_class, per_class_compose):
                img_real = get_images(c, args.batch_real)
                # transform
                # 转换图像
                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                feature_real, output_real = self.local_model(img_real)
                loss_real = criterion(output_real, lab_real)

                # compute the real feature gradients of class c
                # 计算类别 c 的真实特征梯度
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))
                truth_gradient_all[c].append(gw_real)
        for i in list_class:
            gw_real_temp = []
            gradient_all = truth_gradient_all[i]
            weight = 1.0 / len(gradient_all)
            for name_param in range(len(gradient_all[0])):
                list_values_param = []
                for client_one in gradient_all:
                    list_values_param.append(client_one[name_param] * weight)
                value_global_param = sum(list_values_param)
                gw_real_temp.append(value_global_param)
            # the real feature gradients of all classes
            # 所有类别的真实特征梯度
            truth_gradient_avg[i] = gw_real_temp
        return truth_gradient_avg

    def local_train(self, args, global_params, dist):
        # 本地训练函数
        transform_train = transforms.Compose([  # 定义图像预处理操作，包括随即裁剪和水平翻转
            transforms.RandomCrop(32, padding=4),  # 随机裁剪，参数为图像裁剪后的大小和填充大小
            transforms.RandomHorizontalFlip()])  # 随机水平翻转

        # 加载全局参数到本地模型
        self.local_model.load_state_dict(global_params)  # 使用全局参数更新本地模型的参数
        self.local_model.train()  # 将本地模型设置为训练模式

        # 训练若干周期
        for _ in range(args.num_epochs_local_training):  # 迭代训练多个周期
            # 创建数据加载器，用于加载本地客户端的数据批次
            data_loader = DataLoader(dataset=self.data_client,  # 数据为本地客户端的数据
                                     batch_size=args.batch_size_local_training,  # 批次大小为输入大小
                                     shuffle=True)  # 打乱顺序以增加随机性
            for data_batch in data_loader:  # 遍历数据加载器，逐批次进行训练
                images, labels = data_batch  # 获取批次中的图像和标签数据
                images, labels = images.to(self.device), labels.to(self.device)
                images = transform_train(images)

                # FedRS关键
                hs, _ = self.local_model(images)  # 将预处理后的图像输入本地模型进行前向传播，得到预测结果
                ws = self.local_model.classifier.weight
                hs, ws = hs.to(self.device), ws.to(self.device)

                cdist = dist / dist.max()
                cdist = cdist.to(self.device)
                cdist = cdist * (1.0 - args.rs_alpha) + args.rs_alpha
                # 减少极端样本分布对模型训练的干扰
                cdist = cdist.reshape((1, -1))
                # 改变张量的形状，转化为二维的

                logits = cdist * hs.mm(ws.transpose(0, 1))

                # 计算loss
                loss = self.criterion(logits, labels)
                print('loss为：', loss)
                # print(loss.item())
                # 将优化器中之前积累的梯度清零，准备接收新一轮的梯度
                self.optimizer.zero_grad()
                # 反向传播：计算损失函数关于模型参数的梯度
                loss.backward()
                # 根据梯度更新模型参数，执行一步优化
                self.optimizer.step()

        return self.local_model.state_dict()

def FedRS():
    args = args_parser()
    print(
        'imb_factor:{ib}, non_iid:{non_iid}, rs_alpha:{rs_alpha}\n'
        'lr_local_training:{lr_local_training}\n'
        'num_rounds:{num_rounds},num_epochs_local_training:{num_epochs_local_training},batch_size_local_training:{batch_size_local_training}\n'
        'num_online_clients:{num_online_clients}\n'.format(
            ib=args.imb_factor,
            non_iid=args.non_iid_alpha,
            rs_alpha=args.rs_alpha,
            lr_local_training=args.lr_local_training,
            num_rounds=args.num_rounds,
            num_epochs_local_training=args.num_epochs_local_training,
            batch_size_local_training=args.batch_size_local_training,
            num_online_clients=args.num_online_clients))

    random_state = np.random.RandomState(args.seed)

    # Load data 定义数据转化的组合，将图像转为张量，并进行归一化处理
    transform_all = transforms.Compose([
        transforms.ToTensor(),  # 将图片转为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])  # 对图像进行归一化处理
    # 根据命令行参数加载数据集
    if args.dataset == 'cifar10':
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_all)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_all)
    elif args.dataset == 'cifar100':
        data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=transform_all)
        data_global_test = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_all)
    else:
        raise ValueError("Unknown dataset:", args.dataset)

    """
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_all)
    data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_all)
    """
    # Distribute data 分类数据
    # 对数据进行分组，按照类别对数据进行分类 这里用到了dataset里面的函数
    list_label2indices = classify_label(data_local_training, args.num_classes)
    # heterogeneous and long_tailed setting

    # 对数据进行异构和长尾设置
    # 注意：下面的下面的train_long_tail和clients_indices函数需要根据上下文中的实际定义进行解释，因为这些函数在提供的代码中未定义
    a, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                      args.imb_factor, args.imb_type)
    # 根据给定的参数生成客户端的数据索引
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                          args.num_clients, args.non_iid_alpha, args.seed)
    # 展示每个客户端的数据分布情况，并返回原始的数据字典
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes)
    # 全局模型的初始化,其中包括类别数、设备、参数等等
    global_model = Global(num_classes=args.num_classes,
                          device=args.device,
                          args=args,
                          num_of_feature=args.num_of_feature)
    # 创建一个包含所有客户端索引的列表
    total_clients = list(range(args.num_clients))  #
    # 初始化一个“indices2data”，用于将客户端索引映射到数据集
    indices2data = Indices2Dataset(data_local_training)
    re_trained_acc = []
    # 输出多种精确度
    ft_many = []
    ft_medium = []
    ft_few = []
    # 初始化空列表，用于存储重新训练后的准确率

    # 选择一个临时线性模型，用于申城特征和程参数
    temp_model = nn.Linear(256, 10).to(args.device)
    syn_params = temp_model.state_dict()
    # 这个方法会获得一个字典，包含了模型中所有可学习的参数的名称以及对应的张量值。这个字典通常用于保存模型参数或者在模型中传递参数

    # 迭代训练
    for r in tqdm(range(1, args.num_rounds + 1), desc='server-training'):
        """
            作用：遍历从1到args.num_rounds的整数序列，并在每次迭代之前显示一个描述为”server-training“的进度条
            for r in ...: 这是一个 for 循环，它会遍历一个迭代器中的每个元素。循环中的变量 r 用来迭代迭代器中的元素。
            tqdm(...): tqdm 是一个 Python 库，用于在循环中显示进度条，使得长时间运行的循环更具可视化效果。
            range(1, args.num_rounds + 1): 这部分定义了迭代器，它会生成从 1 到 args.num_rounds（包括 args.num_rounds）的整数序列
            desc='server-training': 这个参数设置了进度条的描述文本，显示在进度条前面，用于指示当前循环的目的或阶段。
        """
        # 下载全局参数，global_params为一个字典
        global_params = global_model.download_params()

        # 使用的deepcopy的方法，为了在后续的修改当中，可以安全的对这个修改，而不会影响到原始的global_params
        syn_feature_params = copy.deepcopy(global_params)

        for name_param in reversed(syn_feature_params):
            # 采用了reversed方法，逆序遍历，从而确保在在修改参数的时候不会对后面的参数造成影响
            if name_param == 'classifier.bias':
                # 如果参数名是：classifier.bias，则修改成syn_params['bias'] 中的值
                syn_feature_params[name_param] = syn_params['bias']
            if name_param == 'classifier.weight':
                # 如果参数名是分类器的权重，则修改成syn_params['weight']中的值
                syn_feature_params[name_param] = syn_params['weight']
                break

        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        # random_state.choice是Numoy或者python的一种生成随机数的方法，用于从指定的数组或者列表中随机选择元素
        # 这里的total_clients 是所有用户的索引列表，online_clients 代表的是在线用户的索引列表

        list_clients_gradient = []  # 存储客户端的梯度信息
        list_dicts_local_params = []  # 存储每个客户端的本地参数信息
        list_nums_local_data = []  # 存储每个客户端的本地数量信息
        # local training
        for client in online_clients:

            # FedRS 需要获取当前客户端的分布情况
            cnts = torch.tensor(original_dict_per_client[client])
            # print(f"客户端 {client} 的数据分布: {cnts}")  # 不要忘记转化为张量！
            dist = cnts / cnts.sum()  # 将当前客户端样本分布情况张量中每个类别的样本数量除以总样本数量之和，得到每个类别在当前客户端样本中的相对分布比例。
            # print(f"客户端 {client} 的数据分布: {dist}")

            indices2data.load(list_client2indices[client])
            # 加载指定样本的样本索引集，这里的indices2data在数据集划分的时候就已经出现了
            data_client = indices2data
            # 把这个赋值给data_client 包含特定样本索引的 数据集对象。方便以后操作，不会对原数据造成损坏
            list_nums_local_data.append(len(data_client))
            # 用于记录每个客户端的本地数据集的大小

            local_model = Local(data_client=data_client,
                                class_list=original_dict_per_client[client])
            # local_model 初始化成为了一个local对象，里面有我们预先设定好的初始值，包括优化器，网络什么的
            """
            # compute the real feature gradients in local data
            truth_gradient = local_model.compute_gradient(copy.deepcopy(syn_feature_params), args)
            list_clients_gradient.append(copy.deepcopy(truth_gradient))
            """

            # local update 注意这里更新只用了一部，其余在local里面
            local_params = local_model.local_train(args, copy.deepcopy(global_params), dist)
            # print(local_params)

            # 上面local_params 是经过迭代训练之后产生的本地模型的参数，已经更新完了
            list_dicts_local_params.append(copy.deepcopy(local_params))
            # 把每个本地模型参数的信息保存在list_dicts_local_params这个列表中

        # aggregating local models with FedAvg
        fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
        global_model.update_feature_syn(args, copy.deepcopy(syn_feature_params), list_clients_gradient)
        """
        # re-trained classifier
        syn_params, ft_params = global_model.feature_re_train(copy.deepcopy(fedavg_params),
                                                              args.batch_size_local_training)
        """
        # global eval
        one_re_train_acc = global_model.global_eval(fedavg_params, data_global_test, args.batch_size_test)
        re_trained_acc.append(one_re_train_acc)

        # 输出多种精确度
        many, medium, few = global_model.global_eval_more(fedavg_params, data_global_test, args.batch_size_test, a)
        ft_many.append(many)
        ft_medium.append(medium)
        ft_few.append(few)

        global_model.syn_model.load_state_dict(copy.deepcopy(fedavg_params))
        if r % 10 == 0:
            print("全局精确度：", re_trained_acc)
            print()
            print("多数类的精确度：", ft_many)
            print()
            print("中数类的精确度：", ft_medium)
            print()
            print("少数类的精确度：", ft_few)

    print("全局精确度：", re_trained_acc)
    print()
    print("多数类的精确度：", ft_many)
    print()
    print("中数类的精确度：", ft_medium)
    print()
    print("少数类的精确度：", ft_few)
