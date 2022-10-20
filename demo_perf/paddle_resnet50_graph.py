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

EPOCH_NUM = 3
BATCH_SIZE = 256

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'gpu', 'npu', 'ascend'],
        default="ascend",
        help="Choose the device to run, it can be: cpu/gpu/npu/ascend, default is ascend.")
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
    return parser.parse_args()

# # define a random dataset
# class RandomDataset(paddle.io.Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([3, 224, 224]).astype('float32')
#         label = np.random.randint(0, 9, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

def main(args, place):
    # program
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    with paddle.static.program_guard(main_program, startup_program):
        # model
        model = paddle.vision.models.resnet50()
        cost = nn.CrossEntropyLoss()
        # inputs
        images = static.data(shape=[None, 3, 224, 224], name='image', dtype='float32')
        labels = static.data(shape=[None, 1], name='label', dtype='int64')
        # foward
        outputs = model(images)
        # python/paddle/vision/datasets/folder.py:265
        # import numpy as np
        # return sample, np.array([target]).astype('int64')
        # python/paddle/nn/functional/loss.py:2373
        # label = paddle.unsqueeze(label, axis=axis) # not work in static mode
        loss = cost(outputs, labels)

        # optimizer and amp
        optimizer = paddle.optimizer.SGD(learning_rate=0.1,parameters=model.parameters())
        if args.amp == "O1":
            amp_list = paddle.static.amp.CustomOpLists(
                custom_black_list=["flatten_contiguous_range", "greater_than"])
            optimizer = paddle.static.amp.decorate(
                optimizer=optimizer,
                amp_lists=amp_list,
                init_loss_scaling=1024,
                use_dynamic_loss_scaling=True)

        optimizer.minimize(loss)
        if args.amp == "O1":
            optimizer.amp_init(place, scope=paddle.static.global_scope())

    # # create data loader
    # dataset = RandomDataset(5004 * BATCH_SIZE)
    # train_loader = paddle.io.DataLoader(
    #     dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_loader = paddle.io.DataLoader(
        paddle.vision.datasets.DatasetFolder(
            root='/datasets/ILSVRC2012/train', 
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
        ])),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=32, drop_last=True, prefetch_factor=2)

    # static executor
    exe = static.Executor()
    exe.run(startup_program)

    # switch to train mode
    model.train()
    iter_max = len(train_loader)
    for epoch_id in range(EPOCH_NUM):
        batch_cost = AverageMeter('batch_cost', ':6.3f')
        reader_cost = AverageMeter('reader_cost', ':6.3f')

        # train
        epoch_start = time.time()
        tic = time.time()
        for iter_id, (train_image, train_label) in enumerate(train_loader()):
            # reader_cost
            reader_cost.update(time.time() - tic)

            train_loss = exe.run(
                main_program,
                feed={images.name: train_image,
                      labels.name: train_label},
                fetch_list=[loss])

            # batch_cost and update tic
            batch_cost.update(time.time() - tic)
            tic = time.time()

            # logger for each 100
            if (iter_id+1) % 100 == 0:
                log_info(reader_cost, batch_cost, epoch_id, iter_max, iter_id)        

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
    args = parse_args()
    print('---------------  Running Arguments ---------------')
    print(args)
    print('--------------------------------------------------')

    # enable static and set device
    paddle.enable_static()
    paddle.set_device(args.device)

    
    place = None
    if args.device == "Ascend":
        place = paddle.CustomPlace("Ascend", args.ids)
    elif args.device == "NPU":
        place = paddle.NPUPlace(args.ids)
    elif args.device == "GPU":
        place = paddle.CUDAPlace(args.ids)
    else:
        place = paddle.CPUPlace()
    
    main(args, place)
