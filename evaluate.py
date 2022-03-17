import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tensorboardX import SummaryWriter
from dataset.default import PVDataset
from models.pdvnet import *
import sys
import datetime
import numpy as np


def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='To be filled')
parser.add_argument('--model', default='pvsnet', help='select model')
parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--datapath', help='train datapath')
parser.add_argument('--listfile', help='train list')
parser.add_argument('--num_imgs', default=4, help='number of images')

parser.add_argument('--mode', default='train', help='mode')
parser.add_argument('--gpu', action='store_true', help='enable cuda')
parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="5,10,15:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=4, help='train batch size')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')

parser.add_argument('--summary_freq', type=int, default=1, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

# parse arguments and check
args = parser.parse_args()
if args.resume:
    assert args.loadckpt is None


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# create logger for mode "train" and "testall"
if args.mode == "train":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])

# dataset, dataloader
test_dataset = PVDataset(args.datapath, args.listfile, args.num_imgs, "test")
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = PVNet()
if args.gpu:
    model.cuda()

print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# main function
def evaluate():
    scalars = []
    for batch_idx, sample in enumerate(TestImgLoader):
        scalar = evaluate_sample(sample)

        scalars.append(scalar)


# @make_nograd_func
def evaluate_sample(sample, detailed_summary=True):
    model.eval()
    if args.gpu:
        sample = tocuda(sample)
    scalar = sample["scalar"]
    scalar_est = model(sample["imgs"], sample["proj_mats"], sample["point"], sample["point"])
    return tensor2float(scalar_est)
#



if __name__ == '__main__':
    train()
