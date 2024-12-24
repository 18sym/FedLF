from algorithm.Focal_Loss import *
from algorithm.fedlf import *


if __name__ == '__main__':
    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    random.seed(7)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    args = args_parser()
    if args.algorithm == 'fedlf':
        fedlf()
    else:
        raise ValueError("Unknow algoruithm:", args.algorithm)