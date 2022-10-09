# https://github.com/pytorch/examples/blob/main/imagenet/main.py

import time
import argparse
import datetime

import mindspore as ms
from mindspore import nn
from mindspore import context
import mindvision
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.vision.utils as utils
from mindvision.engine.callback import LossMonitor
from mindvision.classification.models import resnet50

EPOCH_NUM = 3
LOG_STEP = 100
BATCH_SIZE = 256
CALCULATE_DEVICE = "npu:0"
# CALCULATE_DEVICE = "cuda:0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--amp',
        type=str,
        choices=['O0', 'O2', 'O3'],
        default="O2",
        help="Choose the amp level to run, default is O2.")
    parser.add_argument(
        "--graph",
        action='store_true',
        default=False,
        help="Whether to perform graph mode in train")
    return parser.parse_args()

def main():
    args = parse_args()
    print('--------------------------------------------------')
    print(args)
    print('--------------------------------------------------')

    # set device to npu
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    # context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    # model = LeNet5().to(device)
    # network = mindvision.classification.models.resnet50()
    network = resnet50()
    cost = nn.CrossEntropyLoss()
    optimizer = nn.SGD(network.trainable_params(), learning_rate=0.1, momentum=0.9, weight_decay=1e-4)

    # Data loading code
    data_set = ds.ImageFolderDataset('/datasets/ILSVRC2012/train', num_parallel_workers=12, shuffle=True)

    trans = [
        ds.vision.RandomCropDecodeResize(224, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
        ds.vision.RandomHorizontalFlip(prob=0.5)
    ]
    trans_norm = [
        ds.vision.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], 
                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
        ds.vision.HWC2CHW()
    ]
    type_cast_op = ds.transforms.transforms.TypeCast(ms.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=24)
    data_set = data_set.map(operations=trans_norm, input_columns="image", num_parallel_workers=12)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=12)
    data_set = data_set.batch(BATCH_SIZE, drop_remainder=True)

    # mindspore model init
    model = ms.Model(network, loss_fn=cost, optimizer=optimizer, metrics=None, amp_level="O2")

    # start to train
    model.train(EPOCH_NUM, data_set, sink_size=data_set.get_dataset_size(), dataset_sink_mode=True)
    # print_callback = PrintCallBack()
    # model.train(EPOCH_NUM, data_set, callbacks=[print_callback])


# class PrintCallBack(LossMonitor):
#     """
#     Monitor the loss in training.
#     If the loss in NAN or INF terminating training.
#     """

#     def __init__(self, ):
#         super(PrintCallBack, self).__init__()
#         self.batch_cost = AverageMeter('batch_cost', ':6.3f')
#         self.reader_cost = AverageMeter('reader_cost', ':6.3f')

#     def epoch_begin(self, run_context):
#         self.epoch_time = time.time()
#         self.tic = time.time()
#         self.batch_cost.reset()
#         self.reader_cost.reset()

#     def epoch_end(self, run_context):
#         epoch_cost = time.time() - self.epoch_time

#         callback_params = run_context.original_args()
#         epoch_id = callback_params.cur_epoch_num
#         iter_max = callback_params.batch_num
#         avg_ips = iter_max * BATCH_SIZE / epoch_cost
#         print('Epoch ID: {}, Epoch time: {} ms, reader_cost: {:.5f} s, batch_cost: {:.5f} s, reader/batch: {:.2%}, average ips: {:.5f} samples/s'
#             .format(epoch_id+1, epoch_cost * 1000, self.reader_cost.sum, self.batch_cost.sum, self.reader_cost.sum / self.batch_cost.sum, avg_ips))

#     def step_begin(self, run_context):
#         self.reader_cost.update(time.time() - tic)

#     def step_end(self, run_context):
#         batch_cost.update(time.time() - tic)
#         tic = time.time()

#         callback_params = run_context.original_args()
#         epoch_id = callback_params.cur_epoch_num
#         iter_id = callback_params.cur_step_num
#         iter_max = callback_params.batch_num
#         if (iter_id+1) % LOG_STEP == 0:
#                 log_info(reader_cost, batch_cost, epoch_id, iter_max, iter_id)

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
    eta_sec = ((EPOCH_NUM - epoch_id + 1) * iter_max - iter_id) * batch_cost.avg
    eta_msg = "eta: {:s}".format(str(datetime.timedelta(seconds=int(eta_sec))))
    print('Epoch [{}/{}], Iter [{}/{:0>4d}], reader_cost: {:.5f} s, batch_cost: {:.5f} s, ips: {:.5f} samples/s, {}'
          .format(epoch_id+1, EPOCH_NUM, iter_id+1, iter_max, reader_cost.avg, batch_cost.avg, BATCH_SIZE / batch_cost.avg, eta_msg))


if __name__ == '__main__':
    main()
