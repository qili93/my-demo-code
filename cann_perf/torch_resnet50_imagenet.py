# https://github.com/pytorch/examples/blob/main/imagenet/main.py

import time
import argparse
import datetime
import torch
import torch.npu
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from apex import amp

EPOCH_NUM = 3
BATCH_SIZE = 256

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
        default="O1",
        help="Choose the amp level to run, default is O1.")
    parser.add_argument(
        "--graph",
        action='store_true',
        default=False,
        help="Whether to perform graph mode in train")
    return parser.parse_args()

def main():
    args = parse_args()
    print('--------------------------------------------------')
    if args.graph: assert args.device == "Ascend"
    print(args)
    print('--------------------------------------------------')

    # set device to npu
    if args.device == "Ascend":
        CALCULATE_DEVICE = "npu:" + str(args.ids)
        torch.npu.set_device(CALCULATE_DEVICE)
    else:
        CALCULATE_DEVICE = "cuda:" + str(args.ids)

    # model = LeNet5().to(device)
    model = torchvision.models.resnet50().to(CALCULATE_DEVICE)
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # conver to amp model
    if args.amp == "O1" or args.amp == "O2":
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp, loss_scale=1024, verbosity=1)

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('/datasets/ILSVRC2012/train', 
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
        ])),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=32, pin_memory=True, drop_last=True)

    # switch to train mode
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

            # turn on non_blocking when pin_memory is true in data loader
            images = images.to(CALCULATE_DEVICE, non_blocking=True) # high performance with pin_memory and non_blocking
            labels = labels.to(CALCULATE_DEVICE, non_blocking=True) # high performance with pin_memory and non_blocking

            if args.graph:
                torch.npu.enable_graph_mode()
            
            #Forward pass
            outputs = model(images)
            loss = cost(outputs, labels)
                
            # Backward
            if args.amp == "O1" or args.amp == "O2":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            # Optimize
            optimizer.step()
            optimizer.zero_grad()

            if args.graph:
                torch.npu.launch_graph()

            # batch_cost and update tic
            batch_cost.update(time.time() - tic)
            tic = time.time()

            # logger for each 100
            if (iter_id+1) % 100 == 0:
                log_info(reader_cost, batch_cost, epoch_id, iter_max, iter_id)            

        if args.graph:
            torch.npu.disable_graph_mode()
            torch.npu.synchronize()
        
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
    print('Epoch [{}/{}], Iter [{:0>4d}/{}], reader_cost: {:.5f} s, batch_cost: {:.5f} s, exec_cost: {:.5f} s, ips: {:.5f} samples/s, {}'
          .format(epoch_id+1, EPOCH_NUM, iter_id+1, iter_max, reader_cost.avg, batch_cost.avg, batch_cost.avg - reader_cost.avg, BATCH_SIZE / batch_cost.avg, eta_msg))


if __name__ == '__main__':
    main()
