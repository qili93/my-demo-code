# Load in relevant libraries, and alias where appropriate
import time
import argparse
import datetime
import torch
import torch_npu
from torch_npu.npu.amp import GradScaler, autocast
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

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
        action='store_false',
        default=True,
        help="Whether to enable AMP run, default is True.")
    parser.add_argument(
        "--graph",
        action='store_true',
        default=False,
        help="Whether to perform graph mode in train")
    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help='whether to run in debug mode, i.e. run one iter only')
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


def main(args, device):
    # model
    model = LeNet5().to(device=device)
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    # scalar
    scaler = GradScaler(init_scale=128., growth_factor=2.0, enabled=True, growth_interval=1)

    # train
    model.train()
    iter_max = len(train_loader)
    for epoch_id in range(EPOCH_NUM):
        batch_cost = AverageMeter('batch_cost', ':6.3f')
        reader_cost = AverageMeter('reader_cost', ':6.3f')

        epoch_start = time.time()
        tic = time.time()
        for iter_id, (images, labels) in enumerate(train_loader):
            # reader_cost
            reader_cost.update(time.time() - tic)

            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            if args.graph:
                torch.npu.enable_graph_mode()

            if args.amp:
                #Forward pass with autocast if amp
                optimizer.zero_grad()
                with autocast():
                    outputs = model(images)
                    loss = cost(outputs, labels)
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()
                # scaler.step() first unscales gradients of the optimizer's params.
                # If gradients don't contain infs/NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = cost(outputs, labels)
                loss.backward()
                optimizer.step()

            if args.graph:
                torch.npu.launch_graph()

            # batch_cost and update tic
            batch_cost.update(time.time() - tic)
            tic = time.time()

            # log for each step
            log_info(reader_cost, batch_cost, epoch_id, iter_max, iter_id)

            if args.debug and iter_id == 1:
                break

        if args.graph:
            torch.npu.disable_graph_mode()
            torch.npu.synchronize()
        
        if args.debug and epoch_id == 0:
            break

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
        torch_npu.npu.set_device(CALCULATE_DEVICE)
    elif args.device == "GPU":
        CALCULATE_DEVICE = "cuda:" + str(args.ids)
    else:
        CALCULATE_DEVICE = "cpu"
    
    main(args, CALCULATE_DEVICE)
