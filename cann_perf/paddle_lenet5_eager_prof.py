#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import argparse
import datetime
import numpy as np

import paddle
import paddle.nn as nn
import paddle.static as static
import paddle.vision.transforms as transforms
import paddle.profiler as profiler
profiler = profiler.Profiler(targets=[profiler.ProfilerTarget.CUSTOM_DEVICE], custom_device_types=['npu'])

#from line_profiler import LineProfiler

EPOCH_NUM = 5
BATCH_SIZE = 4096

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'gpu', 'npu'],
        default="npu",
        help="Choose the device to run, it can be: cpu/gpu/npu, default is npu.")
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
        '--to_static',
        action='store_true',
        default=False,
        help='whether to enable dynamic to static or not, true or false')
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='whether to run in debug mode, i.e. run one iter only')
    parser.add_argument(
        '--profile',
        action='store_true',
        default=False,
        help='whether to enable ascend profiling or not, true or false')
    return parser.parse_args()


class LeNet5(nn.Layer):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2D(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=0),
            nn.BatchNorm2D(num_features=6),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2D(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0),
            nn.BatchNorm2D(num_features=16),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=2, stride=2))
        self.fc = nn.Linear(in_features=400, out_features=120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = paddle.flatten(out, 1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

def train(args, epoch_id, iter_max, train_loader, model, cost, optimizer, reader_cost, batch_cost, tic):
    if args.profile and epoch_id == 4:
        profiler.start()

    for iter_id, (images, labels) in enumerate(train_loader()):
        # reader_cost
        reader_cost.update(time.time() - tic)
        # forward
        # if args.amp == "O1":
        #     # forward
        #     with paddle.amp.auto_cast(custom_black_list={"flatten_contiguous_range", "greater_than"}, level='O1'):
        #         outputs = model(images)
        #         loss = cost(outputs, labels)
        #     # backward and optimize
        #     scaled = scaler.scale(loss)
        #     scaled.backward()
        #     scaler.minimize(optimizer, scaled)
        # else:
        # forward
        outputs = model(images)
        loss = cost(outputs, labels)
        # backward
        loss.backward()

        # optimize
        optimizer.step()
        optimizer.clear_grad()

        # batch_cost and update tic
        batch_cost.update(time.time() - tic)
        tic = time.time()

        # logger for each step
        log_info(reader_cost, batch_cost, epoch_id, iter_max, iter_id)        

        if args.debug:
            break

    if args.profile and epoch_id == 4:
        profiler.stop()

def main(args):
    # model
    model = LeNet5()

    # cost and optimizer
    cost = nn.CrossEntropyLoss()
    # optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    optimizer = paddle.optimizer.SGD(learning_rate=0.1, parameters=model.parameters())

    # # convert to ampo1 model
    # if args.amp == "O1":
    #     scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    #     model, optimizer = paddle.amp.decorate(models=model, optimizers=optimizer, level='O1')

    # convert to static model
    if args.to_static:
        build_strategy = paddle.static.BuildStrategy()
        model = paddle.jit.to_static(model, build_strategy=build_strategy)

    # data loader
    transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
    ])
    train_loader = paddle.io.DataLoader(
        paddle.vision.datasets.MNIST(mode='train', transform=transform, download=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=32, drop_last=True)


    # switch to train mode
    model.train()
    iter_max = len(train_loader)
    for epoch_id in range(EPOCH_NUM):
        batch_cost = AverageMeter('batch_cost', ':6.3f')
        reader_cost = AverageMeter('reader_cost', ':6.3f')

        # train
        epoch_start = time.time()
        tic = time.time()
        
        # # run with line_profiler
        # profile = LineProfiler()
        # func_wrapped = profile(train)
        # func_wrapped(args, epoch_id, iter_max, train_loader, model, cost, optimizer, reader_cost, batch_cost, tic)
        # profile.print_stats()

        train(args, epoch_id, iter_max, train_loader, model, cost, optimizer, reader_cost, batch_cost, tic)

        epoch_cost = time.time() - epoch_start
        avg_ips = iter_max * BATCH_SIZE / epoch_cost
        print('Epoch ID: {}, Epoch time: {:.5f} s, reader_cost: {:.5f} s, batch_cost: {:.5f} s, exec_cost: {:.5f} s, average ips: {:.5f} samples/s'
            .format(epoch_id+1, epoch_cost, reader_cost.sum, batch_cost.sum, batch_cost.sum - reader_cost.sum, avg_ips))

        if args.debug:
            break


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

    # set device
    paddle.set_device("{}:{}".format(args.device, str(args.ids)))
    
    main(args)
