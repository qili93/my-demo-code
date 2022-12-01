import paddle
import paddle.nn as nn

paddle.set_device("npu")

conv_2d = nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0) # Input: 4,1,28,28 => Output: 4,6,24,24
input = paddle.ones(shape=[4, 1, 28, 28])
output = conv_2d(input)
print(output)
