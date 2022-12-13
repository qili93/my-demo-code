import os
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

paddle.fluid.framework.set_flags({"FLAGS_npu_storage_format": True})

paddle.set_device("npu")

# input = paddle.ones(shape=[4, 1, 28, 28])
# filter = paddle.ones(shape=[6, 1, 5, 5])
# bias = paddle.ones(shape=[6])

input = paddle.ones(shape=[2, 3, 5, 5)
input.stop_gradient = False
# print(f"input={input}")
# input=Tensor(shape=[2, 1, 4, 4], dtype=float32, place=Place(npu:0), stop_gradient=True,
#        [[[[1., 1., 1., 1.],
#           [1., 1., 1., 1.],
#           [1., 1., 1., 1.],
#           [1., 1., 1., 1.]]],


#         [[[1., 1., 1., 1.],
#           [1., 1., 1., 1.],
#           [1., 1., 1., 1.],
#           [1., 1., 1., 1.]]]])

filter = paddle.ones(shape=[6, 3, 3, 3])
filter.stop_gradient = False
# print(f"filter={filter}")
# filter=Tensor(shape=[3, 1, 3, 3], dtype=float32, place=Place(npu:0), stop_gradient=True,
#        [[[[1., 1., 1.],
#           [1., 1., 1.],
#           [1., 1., 1.]]],


#         [[[1., 1., 1.],
#           [1., 1., 1.],
#           [1., 1., 1.]]],


#         [[[1., 1., 1.],
#           [1., 1., 1.],
#           [1., 1., 1.]]]])

out = F.conv2d(x=input, weight=filter, bias=None, stride=1, padding=0, dilation=1, groups=1, data_format='NCHW')

loss = out.sum()

loss.backward()

print("filter.grad=", filter.grad)
print("bias.grad=", bias.grad)
print("input.grad=", input.grad)
