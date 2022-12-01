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
# profiler = profiler.Profiler(targets=[profiler.ProfilerTarget.CUSTOM_DEVICE], custom_device_types=['ascend'])
# profiler = profiler.Profiler(targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU])
profiler = profiler.Profiler(targets=[profiler.ProfilerTarget.CPU])

EPOCH_NUM = 3
BATCH_SIZE = 256

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


# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([1000]).astype('float32')
        label = np.random.randint(0, 1000, (1, )).astype('int32')
        return image, label

    def __len__(self):
        return self.num_samples


def main(args):
    # model
    cost = nn.CrossEntropyLoss()

    # create data loader
    dataset = RandomDataset(5004 * BATCH_SIZE)
    train_loader = paddle.io.DataLoader(dataset, 
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=32, drop_last=True, prefetch_factor=2)

    iter_max = len(train_loader)
    for epoch_id in range(EPOCH_NUM):
        batch_cost = AverageMeter('batch_cost', ':6.3f')
        reader_cost = AverageMeter('reader_cost', ':6.3f')

        # train
        epoch_start = time.time()
        tic = time.time()
        for iter_id, (images, labels) in enumerate(train_loader()):
            # reader_cost
            reader_cost.update(time.time() - tic)


            if args.profile and epoch_id == 1 and iter_id == 2000:
                profiler.start()

            # forward
            loss = cost(images, labels)
            # backward
            loss.backward()

            if args.profile and epoch_id == 1 and iter_id == 2000:
                profiler.stop()
                profiler.summary()

            # batch_cost and update tic
            batch_cost.update(time.time() - tic)
            tic = time.time()

            # logger for each 100 steps
            if (iter_id+1) % 100 == 0:
                log_info(reader_cost, batch_cost, epoch_id, iter_max, iter_id)

            if args.debug:
                break  

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
    print('Epoch [{}/{}], Iter [{:0>4d}/{}], reader_cost: {:.5f} s, batch_cost: {:.5f} s, exec_cost: {:.5f} s, ips: {:.5f} samples/s, {}'
          .format(epoch_id+1, EPOCH_NUM, iter_id+1, iter_max, reader_cost.avg, batch_cost.avg, batch_cost.avg - reader_cost.avg, BATCH_SIZE / batch_cost.avg, eta_msg))


if __name__ == '__main__':
    args = parse_args()
    print('---------------  Running Arguments ---------------')
    print(args)
    print('--------------------------------------------------')

    # set device
    paddle.set_device("{}:{}".format(args.device, str(args.ids)))
    
    main(args)
