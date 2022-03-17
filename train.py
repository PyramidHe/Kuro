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
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

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
#print_args(args)

# dataset, dataloader
train_dataset = PVDataset(args.datapath, args.listfile, args.num_imgs)
test_dataset = PVDataset(args.datapath, args.listfile, args.num_imgs, "test")
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=True, num_workers=4, drop_last=False)

# model, optimizer
model = PVNet()
if args.gpu:
    model.cuda()
model_loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

# load parameters
start_epoch = 0
if args.resume:
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# main function
def train():
    torch.autograd.set_detect_anomaly(True)
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
            print(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TrainImgLoader), loss,
                                                                                     time.time() - start_time))

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))

        # testing
        # avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = test_sample(sample, detailed_summary=do_summary)
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                    len(TestImgLoader), loss,
                                                                                    time.time() - start_time))



def train_sample(sample, detailed_summary=False):
    model.train()
    optimizer.zero_grad()
    if args.gpu:
        sample = tocuda(sample)
    scalar = sample["scalar"]
    #vec = sample["vec"]
    #imgs= sample_cuda["imgs"]
    #vector_est = model(sample_cuda["imgs"], sample_cuda["proj_mats"], sample_cuda["point"])
    scalar_est = model(sample["imgs"], sample["proj_mats"], sample["point"], sample["point"])
    #print("confidence: ", confidence)
    loss = model_loss(scalar_est, scalar)
    loss.backward()
    optimizer.step()

    scalar_outputs = {"loss": loss}
    return tensor2float(loss), tensor2float(scalar_outputs), scalar_est


# @make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()
    if args.gpu:
        sample = tocuda(sample)
    scalar = sample["scalar"]
    scalar_est = model(sample["imgs"], sample["proj_mats"], sample["point"], sample["point"])
    loss = model_loss(scalar_est, scalar)
    scalar_outputs = {"loss": loss}
    return tensor2float(loss), tensor2float(scalar_outputs)
#
#
# def profile():
#     warmup_iter = 5
#     iter_dataloader = iter(TestImgLoader)
#
#     @make_nograd_func
#     def do_iteration():
#         torch.cuda.synchronize()
#         torch.cuda.synchronize()
#         start_time = time.perf_counter()
#         test_sample(next(iter_dataloader), detailed_summary=True)
#         torch.cuda.synchronize()
#         end_time = time.perf_counter()
#         return end_time - start_time
#
#     for i in range(warmup_iter):
#         t = do_iteration()
#         print('WarpUp Iter {}, time = {:.4f}'.format(i, t))
#
#     with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
#         for i in range(5):
#             t = do_iteration()
#             print('Profile Iter {}, time = {:.4f}'.format(i, t))
#             time.sleep(0.02)
#
#     if prof is not None:
#         # print(prof)
#         trace_fn = 'chrome-trace.bin'
#         prof.export_chrome_trace(trace_fn)
#         print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    train()
