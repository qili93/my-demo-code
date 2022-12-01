import time
import datetime

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# profiler
# import paddle.profiler as profiler
# profiler = profiler.Profiler(targets=[profiler.ProfilerTarget.CUSTOM_DEVICE], custom_device_types=['npu'])

EPOCH_NUM = 10
BATCH_NUM = 5000
BATCH_SIZE = 256

def main():
  layer = nn.Sequential(
              nn.Conv2D(in_channels=3, out_channels=6, 
                        kernel_size=5, stride=1, padding=0), # Input: 4,1,28,28 => Output: 4,6,24,24
              nn.BatchNorm2D(num_features=6)) # Input: 4,6,24,24 => Output: 4,6,24,24

  for epoch_id in range(EPOCH_NUM):
    batch_cost = AverageMeter('batch_cost', ':6.3f')
    reader_cost = AverageMeter('reader_cost', ':6.3f')

    epoch_start = time.time()
    tic = time.time()

    for iter_id in range(BATCH_NUM):
      input = paddle.ones(shape=[BATCH_SIZE, 3, 224, 224])
      reader_cost.update(time.time() - tic)

      # if iter_id == 100:
        # profiler.start()

      output = layer(input)

      # if iter_id == 100:
        # profiler.stop()
        # break

      # batch_cost and update tic
      batch_cost.update(time.time() - tic)
      tic = time.time()

      # logger for each step
      if (iter_id+1)%1000 == 0:
        log_info(reader_cost, batch_cost, epoch_id, BATCH_NUM, iter_id)

    epoch_cost = time.time() - epoch_start
    avg_ips = BATCH_NUM * BATCH_SIZE / epoch_cost
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
    # set device
    paddle.set_device("npu")
    main()
