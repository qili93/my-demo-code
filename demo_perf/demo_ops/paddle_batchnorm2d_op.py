import paddle
import paddle.nn as nn
import paddle.nn.functional as F

paddle.set_device("ascend")

layer = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=6, 
                      kernel_size=5, stride=1, padding=0), # Input: 4,1,28,28 => Output: 4,6,24,24
            nn.BatchNorm2D(num_features=6)) # Input: 4,6,24,24 => Output: 4,6,24,24

input = paddle.ones(shape=[4, 1, 28, 28])
output = layer(input)
print(output)
