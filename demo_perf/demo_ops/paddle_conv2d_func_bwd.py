import paddle
import paddle.nn as nn
import paddle.nn.functional as F

paddle.set_device("npu")

input = paddle.ones(shape=[4, 1, 28, 28])
filter = paddle.ones(shape=[6, 1, 5, 5])
bias = paddle.ones(shape=[6])
input.stop_gradient = False
filter.stop_gradient = False
bias.stop_gradient = False

out = F.conv2d(input, filter, bias, padding=1)

loss = out.sum()

loss.backward()


print("filter.grad=", filter.grad)
print("bias.grad=", bias.grad)
print("input.grad=", input.grad)
