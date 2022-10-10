# https://gitee.com/mindspore/models/blob/master/official/cv/resnet/train.py

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
from mindvision.classification.models import resnet50

EPOCH_NUM = 3
LOG_STEP = 100
BATCH_SIZE = 256

DEVICE_ID = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--amp',
        type=str,
        choices=['O0', 'O2', 'O3', 'auto'],
        default="O2",
        help="Choose the amp level to run, default is O2.")
    parser.add_argument(
        "--graph",
        action='store_true',
        default=False,
        help="Whether to perform graph mode in train")
    parser.add_argument(
        '--device',
        type=str,
        choices=['CPU', 'GPU', 'Ascend'],
        default="Ascend",
        help="Choose the device to run, it can be: CPU/GPU/Ascend, default is Ascend.")
    return parser.parse_args()

def main():
    args = parse_args()
    print('--------------------------------------------------')
    print(args)
    print('--------------------------------------------------')

    # set device to npu
    if args.graph:
        context.set_context(mode=context.GRAPH_MODE, device_id=DEVICE_ID, device_target=args.device)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_id=DEVICE_ID, device_target=args.device)

    # model = LeNet5().to(device)
    network = resnet50()
    cost = nn.CrossEntropyLoss()
    optimizer = nn.SGD(network.trainable_params(), learning_rate=0.1, momentum=0.9, weight_decay=1e-4)

    # Data loading code
    data_set = ds.ImageFolderDataset('/datasets/ILSVRC2012/train', num_parallel_workers=12, shuffle=True)

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    trans = [
        ds.vision.RandomCropDecodeResize(224, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
        ds.vision.RandomHorizontalFlip(prob=0.5),
        ds.vision.Normalize(mean=mean, std=std),
        ds.vision.HWC2CHW()
    ]
    type_cast_op = ds.transforms.transforms.TypeCast(ms.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=24)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=12)
    data_set = data_set.batch(BATCH_SIZE, drop_remainder=True)
    step_size = data_set.get_dataset_size()

    # mindspore model init
    model = ms.Model(network, loss_fn=cost, optimizer=optimizer, loss_scale_manager=None, metrics=None, 
        amp_level=args.amp, boost_level="O0", boost_config_dict={"grad_freeze": {"total_steps": EPOCH_NUM * step_size}})

    # start to train
    print_callback = PrintCallBack()
    model.train(EPOCH_NUM, data_set, callbacks=[print_callback],
                sink_size=data_set.get_dataset_size(), dataset_sink_mode=True)


class PrintCallBack(ms.Callback):
    """
    Print time cost and ips when training in mindspore Model train function
    Detail refer to list_callback.on_train_epoch_begin(run_context) in source code
    """

    def __init__(self, ):
        super(PrintCallBack, self).__init__()
        self.batch_cost = AverageMeter('batch_cost', ':6.3f')
        self.reader_cost = AverageMeter('reader_cost', ':6.3f')

    def on_train_epoch_begin(self, run_context):
        self.epoch_time = time.time()
        self.tic = time.time()
        self.batch_cost.reset()
        self.reader_cost.reset()

    def on_train_epoch_end(self, run_context):
        epoch_cost = time.time() - self.epoch_time

        callback_params = run_context.original_args()
        epoch_id = callback_params.cur_epoch_num
        iter_max = callback_params.batch_num
        avg_ips = iter_max * BATCH_SIZE / epoch_cost
        print('Epoch ID: {}, Epoch time: {:.5f} s, reader_cost: {:.5f} s, batch_cost: {:.5f} s, reader/batch: {:.2%}, average ips: {:.5f} samples/s'
            .format(epoch_id, epoch_cost, self.reader_cost.sum, self.batch_cost.sum, self.reader_cost.sum / self.batch_cost.sum, avg_ips))

    def on_train_step_begin(self, run_context):
        self.reader_cost.update(time.time() - self.tic)

    def on_train_step_end(self, run_context):
        self.batch_cost.update(time.time() - self.tic)
        self.tic = time.time()

        callback_params = run_context.original_args()
        epoch_id = callback_params.cur_epoch_num
        iter_max = callback_params.batch_num
        iter_id = callback_params.cur_step_num % iter_max
        if (iter_id+1) % LOG_STEP == 0:
                log_info(self.reader_cost, self.batch_cost, epoch_id, iter_max, iter_id)

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
          .format(epoch_id, EPOCH_NUM, iter_id+1, iter_max, reader_cost.avg, batch_cost.avg, BATCH_SIZE / batch_cost.avg, eta_msg))


if __name__ == '__main__':
    main()
