from torchvision import datasets
# datasets 模块用于，加载和使用预定义的数据集

from torchvision.transforms import ToTensor, transforms
# ToTensor模块用于将PIL图片或者numpy数组转化为Tensor
# transforms模块提供了一系列的图像比那换操作，可用于数据增强

from options import args_parser
# argus_parser，用于解析命令行参数

from Dataset.long_tailed_cifar10 import train_long_tail
# 自己定义的函数，可用于处理长尾数据集CIFAR-10

from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset, TensorDataset, get_class_num
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

from algorithm.fedavg import Fedavg


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
        self.label_syn = torch.tensor([np.ones(self.num_of_feature) * i for i in range(args.num_classes)], dtype=torch.long,
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
        self.feature_net = nn.Linear(256, 100).to(args.device)

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

        # update the federated features.
        # 更新联邦特征
        for ep in range(args.match_epoch):
            loss_feature = torch.tensor(0.0).to(args.device)
            for c in range(args.num_classes):
                if len(gw_real_avg[c]) != 0:
                    feature_syn = self.feature_syn[c * self.num_of_feature:(c + 1) * self.num_of_feature].reshape(
                        (self.num_of_feature, 256))
                    lab_syn = torch.ones((self.num_of_feature,), device=args.device, dtype=torch.long) * c
                    output_syn = self.feature_net(feature_syn)
                    loss_syn = self.criterion(output_syn, lab_syn)

                    # compute the federated feature gradients of class c
                    # 计算类c的联邦特征梯度
                    gw_syn = torch.autograd.grad(loss_syn, net_global_parameters, create_graph=True)
                    loss_feature += match_loss(gw_syn, gw_real_avg[c], args)
            self.optimizer_feature.zero_grad()
            loss_feature.backward()
            self.optimizer_feature.step()

    def feature_re_train(self, fedavg_params, batch_size_local_training):
        # 深度复制特征和程和标签合成
        feature_syn_train_ft = copy.deepcopy(self.feature_syn.detach())
        label_syn_train_ft = copy.deepcopy(self.label_syn.detach())
        # 构建特征-标签训练数据集
        dst_train_syn_ft = TensorDataset(feature_syn_train_ft, label_syn_train_ft)
        # 构建线性模型
        ft_model = nn.Linear(256, 100).to(args.device)
        # 创建优化器
        optimizer_ft_net = SGD(ft_model.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
        ft_model.train()
        # 进行特征重训练
        for epoch in range(args.crt_epoch):
            trainloader_ft = DataLoader(dataset=dst_train_syn_ft,
                                        batch_size=batch_size_local_training,
                                        shuffle=True)
            for data_batch in trainloader_ft:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = ft_model(images)
                loss_net = self.criterion(outputs, labels)
                optimizer_ft_net.zero_grad()
                loss_net.backward()
                optimizer_ft_net.step()
        ft_model.eval()
        # 获取线性模型
        feature_net_params = ft_model.state_dict()
        # 对FedAvg参数进行反向遍历
        for name_param in reversed(fedavg_params):
            if name_param == 'classifier.bias':
                fedavg_params[name_param] = feature_net_params['bias']
            if name_param == 'classifier.weight':
                fedavg_params[name_param] = feature_net_params['weight']
                break
        return copy.deepcopy(ft_model.state_dict()), copy.deepcopy(fedavg_params)

    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        # 初始化fedavg全局参数
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            # 计算全局参数
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        return fedavg_global_params

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
        self.class_compose = class_list    # 类别列表

        self.criterion = CrossEntropyLoss().to(args.device)    # 交叉熵损失函数

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

    def local_train(self, args, global_params):
        # 本地训练函数
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        # 加载全局参数到本地模型
        self.local_model.load_state_dict(global_params)
        self.local_model.train()

        # 训练若干周期
        for _ in range(args.num_epochs_local_training):
            data_loader = DataLoader(dataset=self.data_client,
                                     batch_size=args.batch_size_local_training,
                                     shuffle=True)
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                images = transform_train(images)
                _, outputs = self.local_model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.local_model.state_dict()

def CReFF():
    args = args_parser()
    print(
        'imb_factor:{ib}, non_iid:{non_iid}\n'
        'lr_net:{lr_net}, lr_feature:{lr_feature}, num_of_feature:{num_of_feature}\n '
        'match_epoch:{match_epoch}, re_training_epoch:{crt_epoch}\n'.format(
            ib=args.imb_factor,
            non_iid=args.non_iid_alpha,
            lr_net=args.lr_net,
            lr_feature=args.lr_feature,
            num_of_feature=args.num_of_feature,
            match_epoch=args.match_epoch,
            crt_epoch=args.crt_epoch))
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
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                      args.imb_factor, args.imb_type)
    # 根据给定的参数生成客户端的数据索引
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                          args.num_clients, args.non_iid_alpha, args.seed)
    # 展示每个客户端的数据分布情况，并返回原始的数据字典
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes)
    # 全局模型的初始化
    global_model = Global(num_classes=args.num_classes,
                          device=args.device,
                          args=args,
                          num_of_feature=args.num_of_feature)
    total_clients = list(range(args.num_clients))
    indices2data = Indices2Dataset(data_local_training)
    re_trained_acc = []
    temp_model = nn.Linear(256, 100).to(args.device)
    syn_params = temp_model.state_dict()
    # 迭代训练
    for r in tqdm(range(1, args.num_rounds + 1), desc='server-training'):
        global_params = global_model.download_params()
        syn_feature_params = copy.deepcopy(global_params)
        for name_param in reversed(syn_feature_params):
            if name_param == 'classifier.bias':
                syn_feature_params[name_param] = syn_params['bias']
            if name_param == 'classifier.weight':
                syn_feature_params[name_param] = syn_params['weight']
                break
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_clients_gradient = []
        list_dicts_local_params = []
        list_nums_local_data = []
        # local training
        for client in online_clients:
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            list_nums_local_data.append(len(data_client))
            local_model = Local(data_client=data_client,
                                class_list=original_dict_per_client[client])
            # compute the real feature gradients in local data
            truth_gradient = local_model.compute_gradient(copy.deepcopy(syn_feature_params), args)
            list_clients_gradient.append(copy.deepcopy(truth_gradient))
            # local update
            local_params = local_model.local_train(args, copy.deepcopy(global_params))
            list_dicts_local_params.append(copy.deepcopy(local_params))
        # aggregating local models with FedAvg
        fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
        global_model.update_feature_syn(args, copy.deepcopy(syn_feature_params), list_clients_gradient)
        # re-trained classifier
        syn_params, ft_params = global_model.feature_re_train(copy.deepcopy(fedavg_params),
                                                              args.batch_size_local_training)
        # global eval
        one_re_train_acc = global_model.global_eval(ft_params, data_global_test, args.batch_size_test)
        re_trained_acc.append(one_re_train_acc)
        global_model.syn_model.load_state_dict(copy.deepcopy(fedavg_params))
        if r % 10 == 0:
            print(re_trained_acc)
    print(re_trained_acc)


if __name__ == '__main__':
    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    random.seed(7)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    args = args_parser()
    if args.algorithm == 'creff':
        CReFF()
    elif args.algorithm == 'fedavg':
        Fedavg()
    else:
        raise ValueError("Unknow algoruithm:", args.algorithm)

