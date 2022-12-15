import paddle
import paddle.nn as nn

paddle.set_flags({"FLAGS_npu_storage_format": False})

# paddle.set_device("npu")
paddle.set_device("cpu")

weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0))
bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0))

layer = nn.BatchNorm3D(num_features=4, weight_attr=weight_attr, bias_attr=bias_attr, data_format='NCHW')

input = paddle.ones(shape=[2, 4, 6, 6, 6], dtype='float32')

output = layer(input)

print(output)
