import argparse
"""
    导入argparse模块，使得可以更便捷的多次修改参数，一般分为几个步骤：
    1)导入 argparse模块
    2）创建一个ArgumenParser对象，该对象包含将命令行输入内容解析成Python数据的过程所需的全部功能
        parser = argparse.ArgumentParser()
    3)添加想要输入的命令行参数
        parser.add_argument('radius', type=int, default = (默认的值), help= '')
    4）args = parser.parser_args()  
        ArgumentParser 通过parse_args()方法解析参数，获取命令行中输入的参数
    5）将获取的命令行的输入，当作参数返回到方法中得出结果
    

"""
import os
# 使用os模块，你可以执行诸如文件操作、目录操作等相关的任务。
from Dataset.param_aug import ParamDiffAug
# 自己定义的函数里面的方法

def args_parser():
    parser = argparse.ArgumentParser()
    # 构建对象

    path_dir = os.path.dirname(__file__)
    # 获取当前脚本文所在的目录路径, 此时 path_dir为字符串变量
    # parser.add_argument('--path_inaturalist', type=str, default=os.path.join(path_dir, 'data/iNaturalist18/'))
    # use iNaturalist

    # 通用设置：
    parser.add_argument('--algorithm', type=str, default='creff', choices=['creff', 'fedavg', 'fedprox'],
                        help='choice your algorithm')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_rounds', type=int, default=200,help='全局迭代次数')
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--num_epochs_local_training', type=int, default=10)  #
    parser.add_argument('--batch_size_local_training', type=int, default=32)

    # CREFF
    parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data/CIFAR10/'))
    parser.add_argument('--path_cifar100', type=str, default=os.path.join(path_dir, 'data/CIFAR100/'))
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_online_clients', type=int, default=8)
    parser.add_argument('--match_epoch', type=int, default=100)
    parser.add_argument('--crt_epoch', type=int, default=300)
    parser.add_argument('--batch_real', type=int, default=32)
    parser.add_argument('--num_of_feature', type=int, default=100)
    parser.add_argument('--lr_feature', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_size_test', type=int, default=500)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--non_iid_alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.02, type=float, help='imbalance factor')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--save_path', type=str, default=os.path.join(path_dir, 'result/'))
    parser.add_argument('--method', type=str, default='DSA', help='DC/DSA')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    # FedProx
    parser.add_argument('--mu', type=float, default=0.01)
    # FedAvgM
    (parser
.add_argument('--init_belta', type=float, default=0.97))

    args = parser.parse_args()
    # 把输入的命令行内容返回作为参数，传到方法中获得结果

    # 根据方法名称确定是否开启数据增强
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    return args
