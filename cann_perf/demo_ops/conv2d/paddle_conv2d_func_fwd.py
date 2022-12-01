import paddle
import paddle.nn as nn
import paddle.nn.functional as F

paddle.set_device("npu")

input = paddle.ones(shape=[4, 1, 28, 28])
filter = paddle.ones(shape=[6, 1, 5, 5])
bias = paddle.ones(shape=[6])
out = F.conv2d(input, filter, bias, padding=1)
print(out)
