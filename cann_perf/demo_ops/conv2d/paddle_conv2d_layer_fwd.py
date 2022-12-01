import paddle
import paddle.nn as nn

paddle.set_device("npu")

weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0))
bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0))
conv_2d = nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0,
                    weight_attr=weight_attr, bias_attr=bias_attr) # Input: 4,1,28,28 => Output: 4,6,24,24

input = paddle.ones(shape=[4, 1, 28, 28])
output = conv_2d(input)
print(output)

