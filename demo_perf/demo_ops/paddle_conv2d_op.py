import paddle
import paddle.nn.functional as F

paddle.set_device("ascend")

filters = paddle.ones(shape=[8, 4, 3, 3])
inputs = paddle.ones(shape=[1, 4, 5, 5])
output = F.conv2d(inputs, filters, padding=1)

print(output)


