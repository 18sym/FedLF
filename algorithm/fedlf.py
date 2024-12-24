from Model.log_model import setup_logging
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from options import args_parser
from Dataset.long_tailed_cifar10 import train_long_tail
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset, TensorDataset, \
    get_class_num
from Dataset.sample_dirichlet import clients_indices
import numpy as np
from torch import stack, max, eq, no_grad, tensor, unsqueeze, split
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from Model.Resnet8 import ResNet_cifar
from tqdm import tqdm
import copy
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import torch

def tsne_evaluation(model, dataloader, save_path):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.cuda()
            output, _ = model(inputs)
            features.append(output.cpu().numpy())
            labels.append(targets.numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)

    tsne = TSNE(n_components=2, random_state=0)
    projected_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        indices = labels == i
        plt.scatter(projected_features[indices, 0], projected_features[indices, 1], label=str(i))
    plt.legend()
    plt.savefig(save_path)
    plt.close()

class DecorrLoss(nn.Module):

    def __init__(self):
        super(DecorrLoss, self).__init__()
        self.eps = 1e-8

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        N, C = x.shape
        if N == 1:
            return 0.0

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / N

        return loss

class Global(object):
    def __init__(self,
                 num_classes: int,
                 device: str,
                 args,
                 num_of_feature):
        self.device = device
        self.num_classes = num_classes
        self.fedavg_acc = []
        self.fedavg_many = []
        self.fedavg_medium = []
        self.fedavg_few = []
        self.ft_acc = []
        self.ft_many = []
        self.ft_medium = []
        self.ft_few = []
        self.num_of_feature = num_of_feature
        self.feature_syn = torch.randn(size=(args.num_classes * self.num_of_feature, 256), dtype=torch.float,
                                       requires_grad=True, device=args.device)
        self.label_syn = torch.tensor([np.ones(self.num_of_feature) * i for i in range(args.num_classes)],
                                      dtype=torch.long,
                                      requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
        self.criterion = CrossEntropyLoss().to(args.device)
        self.syn_model = ResNet_cifar(resnet_size=8, scaling=4,
                                      save_activations=False, group_norm_num_groups=None,
                                      freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(device)
        self.feature_net = nn.Linear(256, 10).to(args.device)

    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])

        for name_param in list_dicts_local_params[0]:
            list_values_param = []

            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)

            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param

        return fedavg_global_params


    def global_eval_more(self, fedavg_params, data_test, batch_size_test, a):
        self.syn_model.load_state_dict(fedavg_params)
        self.syn_model.eval()

        majority_threshold = 1500
        # 0.01  0.02
        # minority_threshold = 200
        # 0.1
        minority_threshold = 600

        num_corrects_majority, num_samples_majority = 0, 0
        num_corrects_medium, num_samples_medium = 0, 0
        num_corrects_minority, num_samples_minority = 0, 0
        img_num_class = a
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test, shuffle=False)
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.syn_model(images)
                _, predicts = max(outputs, -1)

                for label, predict in zip(labels, predicts):
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

        accuracy_majority = round(num_corrects_majority / num_samples_majority, 4) if num_samples_majority > 0 else 0
        accuracy_medium = round(num_corrects_medium / num_samples_medium, 4) if num_samples_medium > 0 else 0
        accuracy_minority = round(num_corrects_minority / num_samples_minority, 4) if num_samples_minority > 0 else 0

        return accuracy_majority, accuracy_medium, accuracy_minority

    def global_eval(self, fedavg_params, data_test, batch_size_test):
        self.syn_model.load_state_dict(fedavg_params)
        self.syn_model.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.syn_model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return self.syn_model.state_dict()

class Local(object):
    def __init__(self,
                 data_client,
                 class_list: int):
        args = args_parser()
        self.data_client = data_client
        self.device = args.device
        self.class_compose = class_list
        self.criterion = CrossEntropyLoss().to(args.device)  # 交叉熵损失函数
        self.feddecorr = DecorrLoss()
        self.local_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(args.device)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)

    def local_train(self, args, global_params, dist):

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        self.local_model.load_state_dict(global_params)
        self.local_model.train()

        feature_centers = torch.zeros((args.num_classes, 256), device=self.device)
        class_counts = torch.zeros(args.num_classes, device=self.device)

        pre_loader = DataLoader(dataset=self.data_client, batch_size=args.batch_size_local_training, shuffle=True)
        with torch.no_grad():
            for images, labels in pre_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                images = transform_train(images)
                features, _ = self.local_model(images)
                for i in range(features.size(0)):
                    feature_centers[labels[i]] += features[i]
                    class_counts[labels[i]] += 1

            valid_classes = class_counts != 0
            feature_centers[valid_classes] /= class_counts[valid_classes].unsqueeze(1)
            feature_centers[~valid_classes] = 1e-8
            # torch.randn(1, feature_centers.size(1), device=self.device)

            gap = torch.ones((args.num_classes, args.num_classes), device=self.device) * 1e9
            for i in range(args.num_classes):
                for j in range(i):
                    dis = torch.norm(feature_centers[i] - feature_centers[j], p=2)
                    gap[i, j] = dis
                    gap[j, i] = dis
            min_gap = torch.min(gap[torch.triu(torch.ones_like(gap), diagonal=1) > 0])
            max_gap = torch.max(gap[torch.triu(torch.ones_like(gap), diagonal=1) > 0])
            # print('class-wise minimum distance:', gap)
            # print('min_gap:', min_gap)
            # print('max_gap:', max_gap)
            # print('max_gap.item():', max_gap.item())

        # 训练若干周期
        for _ in range(args.num_epochs_local_training):
            data_loader = DataLoader(dataset=self.data_client,
                                     batch_size=args.batch_size_local_training,
                                     shuffle=True)
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                images = transform_train(images)
                hs, _ = self.local_model(images)
                ws = self.local_model.classifier.weight
                hs, ws = hs.to(self.device), ws.to(self.device)

                cdist = dist / dist.max()
                cdist = cdist.to(self.device)
                cdist = cdist * (1.0 - args.rs_alpha) + args.rs_alpha
                cdist = cdist.reshape((1, -1))
                logits = cdist * hs.mm(ws.transpose(0, 1))
                loss1 = self.criterion(logits, labels)

                features_square = torch.sum(torch.pow(hs, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(feature_centers, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(hs, feature_centers.T)
                dist_2 = features_square - 2 * features_into_centers + centers_square.T
                dist_2 = torch.sqrt(dist_2)
                one_hot = F.one_hot(labels, args.num_classes).to(self.device)
                gap = min(max_gap.item(), 100)
                dist_2 = dist_2 + one_hot * gap
                loss2 = self.criterion(-dist_2, labels)
                # print('loss_2:', loss2)
                loss_decorr = self.feddecorr(hs)
                loss = loss1 + loss2 * 0.01 + loss_decorr * 0.01
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self.local_model.state_dict()

def fedlf():
    logger = setup_logging('50-fedlf_log_file.log')
    args = args_parser()
    logger.info(
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

    # 别忘了修改阈值
    if args.imb_factor == 0.01:
        logger.info("majority_threshold: {}".format(1500))
        logger.info("minority_threshold: {}".format(200))
    elif args.imb_factor == 0.02:
        logger.info("majority_threshold: {}".format(1500))
        logger.info("minority_threshold: {}".format(200))
    elif args.imb_factor == 0.1:
        logger.info("majority_threshold: {}".format(1500))
        logger.info("minority_threshold: {}".format(600))

    random_state = np.random.RandomState(args.seed)


    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_all)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_all)
    elif args.dataset == 'cifar100':
        data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=transform_all)
        data_global_test = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_all)
    else:
        raise ValueError("Unknown dataset:", args.dataset)

    list_label2indices = classify_label(data_local_training, args.num_classes)
    a, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                      args.imb_factor, args.imb_type)
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                          args.num_clients, args.non_iid_alpha, args.seed)
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes)
    global_model = Global(num_classes=args.num_classes,
                          device=args.device,
                          args=args,
                          num_of_feature=args.num_of_feature)
    total_clients = list(range(args.num_clients))  #
    indices2data = Indices2Dataset(data_local_training)
    re_trained_acc = []
    ft_many = []
    ft_medium = []
    ft_few = []
    temp_model = nn.Linear(256, 10).to(args.device)
    syn_params = temp_model.state_dict()

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

            cnts = torch.tensor(original_dict_per_client[client])
            dist = cnts / cnts.sum()
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            list_nums_local_data.append(len(data_client))

            local_model = Local(data_client=data_client,
                                class_list=original_dict_per_client[client])
            local_params = local_model.local_train(args, copy.deepcopy(global_params), dist)
            # print(local_params)
            list_dicts_local_params.append(copy.deepcopy(local_params))

        # aggregating local models with FedAvg
        fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
        global_model.update_feature_syn(args, copy.deepcopy(syn_feature_params), list_clients_gradient)

        # global eval
        one_re_train_acc = global_model.global_eval(fedavg_params, data_global_test, args.batch_size_test)
        re_trained_acc.append(one_re_train_acc)

        many, medium, few = global_model.global_eval_more(fedavg_params, data_global_test, args.batch_size_test, a)
        ft_many.append(many)
        ft_medium.append(medium)
        ft_few.append(few)

        global_model.syn_model.load_state_dict(copy.deepcopy(fedavg_params))
        if r % 10 == 0:
            logger.info("Global Accuracy: {}".format(re_trained_acc))
            print()
            logger.info("Majority Class Accuracy: {}".format(ft_many))
            print()
            logger.info("Medium Class Accuracy: {}".format(ft_medium))
            print()
            logger.info("Minority Class Accuracy: {}".format(ft_few))

            test_loader = DataLoader(data_global_test, batch_size=args.batch_size_test, shuffle=False)
            save_dir = "Dimensionality_reduction/vsloss_feature"
            os.makedirs(save_dir, exist_ok=True)  # 确保目录存在，不存在则创建
            save_path = os.path.join(save_dir, f"tsne_epoch_{r}.png")
            tsne_evaluation(global_model.syn_model, test_loader, save_path=save_path)

    logger.info("Global Accuracy: {}".format(re_trained_acc))
    print()
    logger.info("Majority Class Accuracy: {}".format(ft_many))
    print()
    logger.info("Majority Class Accuracy: {}".format(ft_medium))
    print()
    logger.info("Majority Class Accuracy: {}".format(ft_few))

