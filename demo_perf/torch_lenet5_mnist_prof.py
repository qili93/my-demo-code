# Load in relevant libraries, and alias where appropriate
import time
import argparse
import datetime
import torch
import torch.npu
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from apex import amp

from line_profiler import LineProfiler

EPOCH_NUM = 5
BATCH_SIZE = 4096

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        type=str,
        choices=['CPU', 'GPU', 'Ascend'],
        default="Ascend",
        help="Choose the device to run, it can be: CPU/GPU/Ascend, default is Ascend.")
    parser.add_argument(
        '--ids',
        type=int,
        default=0,
        help="Choose the device id to run, default is 0.")
    parser.add_argument(
        '--amp',
        type=str,
        choices=['O0', 'O1', 'O2'],
        default="O0",
        help="Choose the amp level to run, default is O1.")
    parser.add_argument(
        "--graph",
        action='store_true',
        default=False,
        help="Whether to perform graph mode in train")
    return parser.parse_args()


#Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


def train_func(args, epoch_id, iter_max, train_loader, model, cost, optimizer, reader_cost, batch_cost, tic, device):
    for iter_id, (images, labels) in enumerate(train_loader):
            # reader_cost
            reader_cost.update(time.time() - tic)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # if args.graph:
            #     torch.npu.enable_graph_mode()
            
            #Forward pass
            outputs = model(images)
            loss = cost(outputs, labels)
                
            # # Backward and optimize
            # if args.amp == "O1" or args.amp == "O2":
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()

            # Optimize
            optimizer.step()
            optimizer.zero_grad()

            # if args.graph:
            #     torch.npu.launch_graph()

            # batch_cost and update tic
            batch_cost.update(time.time() - tic)
            tic = time.time()

            # log for each step
            log_info(reader_cost, batch_cost, epoch_id, iter_max, iter_id)

            # for debug
            # break

def main(args, device):
    # model
    model = LeNet5().to(device)
    cost = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # # conver to amp model
    # if args.amp == "O1" or args.amp == "O2":
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp, loss_scale=1024, verbosity=1)

    # data loader
    transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.1307,), std = (0.3081,))])

    train_loader = torch.utils.data.DataLoader(
        dataset = torchvision.datasets.MNIST(root = './data', train = True, 
            transform = transform, download = True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=32, pin_memory=True, drop_last=True)


    # train
    model.train()
    iter_max = len(train_loader)
    for epoch_id in range(EPOCH_NUM):
        batch_cost = AverageMeter('batch_cost', ':6.3f')
        reader_cost = AverageMeter('reader_cost', ':6.3f')

        epoch_start = time.time()
        tic = time.time()


        # run with line_profiler
        profile = LineProfiler()
        func_wrapped = profile(train_func)
        func_wrapped(args, epoch_id, iter_max, train_loader, model, cost, optimizer, reader_cost, batch_cost, tic, device)
        profile.print_stats()

        # train_func(args, epoch_id, iter_max, train_loader, model, cost, optimizer, reader_cost, batch_cost, tic, device)

        # if args.graph:
        #     torch.npu.disable_graph_mode()
        #     torch.npu.synchronize()
        
        epoch_cost = time.time() - epoch_start
        avg_ips = iter_max * BATCH_SIZE / epoch_cost
        print('Epoch ID: {}, Epoch time: {:.5f} s, reader_cost: {:.5f} s, batch_cost: {:.5f} s, exec_cost: {:.5f} s, average ips: {:.5f} samples/s'
            .format(epoch_id+1, epoch_cost, reader_cost.sum, batch_cost.sum, batch_cost.sum - reader_cost.sum, avg_ips))


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Code was based on https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name='', fmt='f', postfix="", need_avg=True):
        self.name = name
        self.fmt = fmt
        self.postfix = postfix
        self.need_avg = need_avg
        self.reset()

    def reset(self):
        """ reset """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ update """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_info(reader_cost, batch_cost, epoch_id, iter_max, iter_id):
    eta_sec = ((EPOCH_NUM - epoch_id) * iter_max - iter_id) * batch_cost.avg
    eta_msg = "eta: {:s}".format(str(datetime.timedelta(seconds=int(eta_sec))))
    print('Epoch [{}/{}], Iter [{:0>2d}/{}], reader_cost: {:.5f} s, batch_cost: {:.5f} s, exec_cost: {:.5f} s, ips: {:.5f} samples/s, {}'
          .format(epoch_id+1, EPOCH_NUM, iter_id+1, iter_max, reader_cost.avg, batch_cost.avg, batch_cost.avg - reader_cost.avg, BATCH_SIZE / batch_cost.avg, eta_msg))


if __name__ == '__main__':
    args = parse_args()
    print('---------------  Running Arguments ---------------')
    print(args)
    print('--------------------------------------------------')
    
    CALCULATE_DEVICE = None
    if args.device == "Ascend":
        CALCULATE_DEVICE = "npu:" + str(args.ids)
        torch.npu.set_device(CALCULATE_DEVICE)
    elif args.device == "GPU":
        CALCULATE_DEVICE = "cuda:" + str(args.ids)
    else:
        CALCULATE_DEVICE = "cpu"

    # run with line_profiler
    # profile = LineProfiler()
    # func_wrapped = profile(main)
    # func_wrapped(args, CALCULATE_DEVICE)
    # profile.print_stats()

    main(args, CALCULATE_DEVICE)
