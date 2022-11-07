import paddle
import paddle.nn as nn
import paddle.nn.functional as F

paddle.set_device("ascend")

# filters = paddle.ones(shape=[8, 4, 3, 3])
# inputs = paddle.ones(shape=[1, 4, 5, 5])
# output = F.conv2d(inputs, filters, padding=1)
# print(output)

conv_2d = nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0) # Input: 4,1,28,28 => Output: 4,6,24,24
input = paddle.ones(shape=[4, 1, 28, 28])
output = conv_2d(input)
print(output)

# bias_tensor = paddle.nn.utils.parameters_to_vector(conv_2d.bias)
# bias_tensor = paddle.npu_identity(bias_tensor, 3)
# paddle.nn.utils.vector_to_parameters([bias_tensor], conv_2d.bias)

# for name, param in conv_2d.named_parameters():
#     print(name, param)

