#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import moxing as mox
import datetime
import numpy as np

import paddle
import paddle.nn as nn
import paddle.static as static
import paddle.vision.transforms as transforms

EPOCH_NUM = 5
BATCH_SIZE = 4096

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_url',
        type=str,
        default='./data',
        help="fixed obs data input url in modelarts")
    parser.add_argument(
        '--train_url',
        type=str,
        default='./model',
        help="fixed obs train output url in modelarts")
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
        default="O1",
        help="Choose the amp level to run, default is O1.")
    parser.add_argument(
        '--to_static',
        action='store_true',
        default=False,
        help='whether to enable dynamic to static or not, true or false')
    return parser.parse_args()


def test(epoch_id, test_loader, model, cost):
    model.eval()
    avg_acc = [[], []]
    for images, labels in test_loader():
        outputs = model(images)
        acc_top1 = paddle.metric.accuracy(input=outputs, label=labels, k=1)
        acc_top5 = paddle.metric.accuracy(input=outputs, label=labels, k=5)
        avg_acc[0].append(acc_top1.numpy())
        avg_acc[1].append(acc_top5.numpy())
    model.train()
    top1_avg = np.array(avg_acc[0]).mean()
    top5_avg = np.array(avg_acc[1]).mean()
    print(f"Epoch ID: {epoch_id + 1}, "
          f"Top1 accurary:: {top1_avg:.5f}, "
          f"Top5 accurary:: {top5_avg:.5f}")
    return top1_avg, top5_avg


def main(args):
    model = paddle.vision.models.LeNet()
    cost = nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=model.parameters())

    # convert to ampo1 model
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    model, optimizer = paddle.amp.decorate(
        models=model, optimizers=optimizer, level='O1')

    # data loader
    transform = transforms.Compose([
        transforms.Resize((28, 28)), transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307, ), std=(0.3081, ))
    ])

    train_loader = paddle.io.DataLoader(
        paddle.vision.datasets.MNIST(
            image_path=args.data_url+"/train-images-idx3-ubyte.gz",
            label_path=args.data_url+"/train-labels-idx1-ubyte.gz",
            mode='train', transform=transform),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=32, drop_last=True)

    test_loader = paddle.io.DataLoader(
        paddle.vision.datasets.MNIST(
            image_path=args.data_url+"/t10k-images-idx3-ubyte.gz",
            label_path=args.data_url+"/t10k-labels-idx1-ubyte.gz",
            mode='test', transform=transform),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=32, drop_last=True)

    # switch to train mode
    model.train()
    iter_max = len(train_loader)
    for epoch_id in range(EPOCH_NUM):
        batch_cost = AverageMeter('batch_cost', ':6.3f')

        # train
        epoch_start = time.time()
        tic = time.time()
        for iter_id, (images, labels) in enumerate(train_loader()):
            # forward
            with paddle.amp.auto_cast(
                    custom_black_list={
                        "flatten_contiguous_range", "greater_than"
                    },
                    level='O1'):
                outputs = model(images)
                loss = cost(outputs, labels)
            # backward and optimize
            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.minimize(optimizer, scaled)
            optimizer.clear_grad()

            # batch_cost and update tic
            batch_cost.update(time.time() - tic)
            tic = time.time()

            # logger for each step
            log_info(epoch_id, iter_id, iter_max, batch_cost, loss.numpy()[0])

        epoch_cost = time.time() - epoch_start
        avg_ips = iter_max * BATCH_SIZE / epoch_cost
        print(f"Epoch ID: {epoch_id + 1}, Epoch time: {epoch_cost:>5f} s, "
              f"loss: {loss.numpy()[0]:>7f}, "
              f"average ips: {avg_ips:>5f} samples/s")

        # evaluate after each epoch
        top1_avg, top5_avg = test(epoch_id, test_loader, model, cost)
        metric_info = {"epoch": epoch_id + 1,
                       "loss": loss.numpy()[0], 
                       "top1": top1_avg,
                       "top5": top5_avg}

        # save model to checkpoint
        model_path = args.train_url + "/epoch_" + str(epoch_id + 1)
        para_dict = model.state_dict()
        para_dict.update(cost.state_dict())
        opti_dict = optimizer.state_dict()
        paddle.save(para_dict, model_path + ".pdparams")
        paddle.save(opti_dict, model_path + ".pdopt")
        paddle.save(metric_info, model_path+ ".pdstates")

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


def log_info(epoch_id, iter_id, iter_max, batch_cost, loss):
    eta_sec = ((EPOCH_NUM - epoch_id) * iter_max - iter_id) * batch_cost.avg
    eta_msg = "eta: {:s}".format(str(datetime.timedelta(seconds=int(eta_sec))))
    print(f"Epoch [{epoch_id + 1}/{EPOCH_NUM}], "
          f"Iter [{iter_id + 1:0>2d}/{iter_max}], "
          f"Iter time: {batch_cost.avg:>5f} s, "
          f"loss: {loss:>7f}, "
          f"ips: {BATCH_SIZE /batch_cost.avg:>5f} samples/s, {eta_msg}")

if __name__ == '__main__':
    args = parse_args()

    # download dataset from obs
    obs_data_url = args.data_url
    args.data_url = '/home/work/user-job-dir/inputs/data/'
    obs_train_url = args.train_url
    args.train_url = '/home/work/user-job-dir/outputs/model/'
    try:
        mox.file.copy_parallel(obs_data_url, args.data_url)
        print(f"Successfully Download {obs_data_url} to {args.data_url}")
    except Exception as e:
        print(f"moxing download {obs_data_url} to {args.data_url} failed: {str(e)}")

    paddle.set_device("{}:{}".format(args.device, str(args.ids)))
    main(args)

    # upload model checkpoint to obs
    try:
        mox.file.copy_parallel(args.train_url, obs_train_url)
        print(f"Successfully Upload {args.train_url} to {obs_train_url}")
    except Exception as e:
        print(f"moxing upload {args.train_url} to {obs_train_url} failed: {str(e)}")
