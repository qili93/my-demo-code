import paddle
import paddle.nn as nn
import paddle.nn.functional as F

paddle.set_device("npu")

x = paddle.rand([])
y = paddle.nn.functional.relu(x)

print("x=", x)
print("y=", y)
