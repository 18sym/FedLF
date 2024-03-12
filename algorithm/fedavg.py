import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from options import args_parser
from Model.CNN import CNNCifar
from Dataset.dataset import *
from Dataset.long_tailed_cifar10 import *
from Dataset.sample_dirichlet import *
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        # 初始方法，接受原始数据对象和要包含的索引列表
        self.dataset = dataset  # 将原始数据对象赋值给slef.dataset
        self.idxs = list(idxs)  # 将传入的索引列表转化为列表，并赋值给slef.idxs

    def __len__(self):
        return len(self.idxs)  # 输出列表的长度，也就是数据量的的多少

    def __getitem__(self, item):
        # 获取特定索引item处的样本
        image, label = self.dataset[self.idxs[item]]

        return image, label

class LocalUpdate(object):
    def __init__(self,args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size_local_training, shuffle=True)
        # DatasetSplit(dataset, idxs) 返回的是image,label

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr_local_training, momentum=self.args.momentum)

        epoch_loss = []  # 定义了一个空列表，每一个epoch结束之后，用于存储每一个epoch的值，为了更好的观察损失的变化
        # 在每个本地训练当中轮次进行迭代
        for iter in range(self.args.num_epochs_local_training):
            batch_loss = []  # 初始化一个空列表，用于存储每个批次的损失值
            # print(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # 遍历本地数据加载器中的每个批次，这里使用batch_idx是因为更适合，所以中间变量的取值最好要贴近现实
                images, labels = images.to(torch.device('cuda')), labels.to(torch.device('cuda'))
                # 将图像和标签移动到指定的设备（通常是 GPU）
                net.zero_grad()
                # 清零网络的梯度 在每次迭代中，都需要首先将之前的梯度清零，以防止梯度累积。这就是使用 zero_grad() 方法的原因。
                log_probs = net(images)
                # 通过网络进行前向传播，获取对数概率，也就是将图片输入到模型当中，产生的值，通常情况下是模型对于每个类型预测概率的对数值
                loss = self.loss_func(log_probs, labels)
                # 计算损失值
                loss.backward()  # 对loss执行反向传播算法
                # 反向传播，计算梯度
                optimizer.step()
                """
                    更新模型参数调用 optimizer.step() 方法后，模型中的参数值将会被更新，以反映损失函数梯度下降的方向。
                    这样，通过反复迭代前向传播、损失计算、反向传播和参数更新的过程，
                    神经网络模型将逐渐优化，以更好地适应训练数据并提高性能。
                """
                # print(batch_idx)
                if batch_idx % 10 == 0:
                    # 如果设置了详细模式并且当前批次的索引能够被 10 整除，就打印训练进度信息
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            # 将当前批次的损失值添加到列表中,注意是batch_loss not epoch_loss
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # 将当前迭代轮次的平均损失值添加到列表中
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
        # 注意 这里模型的返回值依然是字典，调用了state_dict()方法
        # 返回训练后的网络参数和平均损失值


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss




def fedavg(w):
    # 创建一个深度拷贝，用于存储计算得到的平均模型参数
    w_avg = copy.deepcopy(w[0])
    # 遍历模型参数的键，因为模型参数为一个字典
    for k in w_avg.keys():
        for i in range(1, len(w)):
            # 对应模型的键值相加
            w_avg[k] += w[i][k]
        # 计算平均值，即对应键的模型参数之和除以客户端的数量
        # 因为w_locals[idx] = copy.deepcopy(w)，所以len（w）就是客户端的数量
        w_avg[k] = torch.div(w_avg[k], len(w))
    # 返回计算得到的平均模型参数，为一个字典
    return w_avg

def Fedavg():
    args = args_parser()
    # 设置设备使用
    print(
        'imb_factor:{ib}, non_iid:{non_iid}\n'
        'dataset:{dataset}, num_clients:{num_clients}\n'
        'num_rounds:{num_rounds}, num_epochs_local_training:{num_epochs_local_training}, batch_size_local_training:{batch_size_local_training}\n'
        'lr_local_training:{lr_local_training}\n'.format(
            ib=args.imb_factor,
            non_iid=args.non_iid_alpha,
            dataset=args.dataset,
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            num_epochs_local_training=-args.num_epochs_local_training,
            batch_size_local_training=args.batch_size_local_training,
            lr_local_training=args.lr_local_training))

    random_state = np.random.RandomState(args.seed)

    args.device = torch.device('cuda')

    # 加载数据集
    if args.dataset == 'cifar10':
        trans_cifar10 = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
        dataset_train = datasets.CIFAR10(root='../data/CIFAR10', train=True, download=True, transform=trans_cifar10)
        dataset_test = datasets.CIFAR10(root='../data/CIFAR10', train=False, download=True, transform=trans_cifar10)

        # Distribute data 分类数据
        # 对数据进行分组，按照类别对数据进行分类 这里用到了dataset里面的函数
        list_label2indices = classify_label(dataset_train, 10)
        # heterogeneous and long_tailed setting

        # 对数据进行异构和长尾设置
        # 注意：下面的下面的train_long_tail和clients_indices函数需要根据上下文中的实际定义进行解释，因为这些函数在提供的代码中未定义
        _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), 10,
                                                          args.imb_factor, args.imb_type)
        # 根据给定的参数生成客户端的数据索引
        list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), 10,
                                              args.num_clients, args.non_iid_alpha, args.seed)
        # 展示每个客户端的数据分布情况，并返回原始的数据字典
        original_dict_per_client = show_clients_data_distribution(dataset_train, list_client2indices,
                                                                  args.num_classes)

    # model
    net_glob = CNNCifar(args=args).to(args.device)
    print(net_glob)
    net_glob.train()

    # copy weight
    w_glob = net_glob.state_dict()
    # 这里将模型转化为字典后赋值给了W_glob

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    w_locals = [w_glob for i in range(args.num_clients)]
    # 格式 [{},{},{}]

    for iter in range(args.num_rounds):  # 对于每一轮迭代
        loss_locals = []
        # 计算要参与训练的客户端数量
        m = max(int(args.frac * args.num_clients), 1)
        # 从客户端列表中随机选择 m 个客户端进行训练 idxs_users是数组[2, 7, 9]
        idxs_users = np.random.choice(range(args.num_clients), m, replace=False)

        # 对于每个选择的客户端索引
        for idx in idxs_users:  # idx 为中间变量，什么值都可以，循环结束的时候idx也就消失了，你也可以使用_代表中间变量
            dict_users = {idx: set(indices) for idx, indices in enumerate(list_client2indices)}
            # print(dict_users)
            local = LocalUpdate(args=args,dataset=dataset_train, idxs=dict_users[idx])
            # 本地训练之后返回两个值net.state_dict()：字典 当前epoch的平均损失sum(epoch_loss) / len(epoch_loss)
            # 在选定的客户端上执行本地训练，并获取更新后的模型和损失
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            # 因为train所需要的参数就是模型
            w_locals[idx] = copy.deepcopy(w)
            # 将本地损失添加到损失列表中，之前定义为空的列表
            loss_locals.append(copy.deepcopy(loss))

        w_glob = fedavg(w_locals)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        # net_glob.load_state_dict(w_glob) 方法，可以将 net_glob 的参数加载为 w_glob 中存储的参数。
        # 这样做的目的是将全局模型恢复为先前训练好的状态，或者在联邦学习中用于更新全局模型。

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)


# testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

"""        
        # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
"""