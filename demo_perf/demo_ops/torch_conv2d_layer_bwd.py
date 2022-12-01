# reference: https://johnwlambert.github.io/conv-backprop/

import torch
import torch_npu
import torch.npu
import numpy as np


# ==== Identity ==== ??
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6, 1, 5, 5], StorageShape = [6, 1, 5, 5], shapeRange = [], memtype = 0, isConst = 0
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 4, Shape = [6, 1, 5, 5], StorageShape = [25, 1, 16, 16], shapeRange = [], memtype = 0, isConst = 0


 # Input: 4,1,28,28 => Output: 4,6,24,24
conv_2d = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0).to("npu:0")
conv_2d.weight = torch.nn.Parameter(torch.ones(6, 1, 5, 5).to("npu:0"))
conv_2d.bias = torch.nn.Parameter(torch.ones(6).to("npu:0"))

x_tensor = torch.ones(4, 1, 28, 28).to("npu:0")
x_tensor.requires_grad = True

# ==== Conv2D ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 1, 28, 28], StorageShape = [4, 1, 28, 28], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6, 1, 5, 5], StorageShape = [6, 1, 5, 5], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[2]: [TensorDesc] DataType = 0, Format = 2, StorageFormat = 2, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [4, 6, 24, 24], StorageShape = [4, 1, 24, 24, 16], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {data_format = NCHW, dilations = [1, 1, 1, 1], groups = 1, pads = [0, 0, 0, 0], strides = [1, 1, 1, 1]}
out = conv_2d(x_tensor)

# ==== ReduceSum ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [4, 6, 24, 24], StorageShape = [4, 1, 24, 24, 16], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 9, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 2, StorageFormat = 2, Shape = [], StorageShape = [], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {keep_dims = False}
loss = out.mean()

# ==== OnesLike ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 2, StorageFormat = -1, Shape = [], StorageShape = [], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 2, StorageFormat = 2, Shape = [], StorageShape = [], shapeRange = [], memtype = 0, isConst = 0

# ==== BroadcastTo ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [1, 1, 1, 1], StorageShape = [1, 1, 1, 1], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 9, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 6, 24, 24], StorageShape = [4, 6, 24, 24], shapeRange = [], memtype = 0, isConst = 0

# ==== Conv2DBackpropInput ====
# InputDesc[0]: [TensorDesc] DataType = 3, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6, 1, 5, 5], StorageShape = [6, 1, 5, 5], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[2]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 6, 24, 24], StorageShape = [4, 6, 24, 24], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [4, 1, 28, 28], StorageShape = [4, 1, 28, 28, 16], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {data_format = NCHW, dilations = [1, 1, 1, 1], groups = 1, pads = [0, 0, 0, 0], strides = [1, 1, 1, 1]}

# ==== BroadcastTo ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [1, 1, 1, 1], StorageShape = [1, 1, 1, 1], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 9, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 6, 24, 24], StorageShape = [4, 6, 24, 24], shapeRange = [], memtype = 0, isConst = 0

# ==== Conv2DBackpropFilter ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 1, 28, 28], StorageShape = [4, 1, 28, 28], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 3, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# InputDesc[2]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 6, 24, 24], StorageShape = [4, 6, 24, 24], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 4, Shape = [6, 1, 5, 5], StorageShape = [25, 1, 16, 16], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {data_format = NCHW, dilations = [1, 1, 1, 1], groups = 1, pads = [0, 0, 0, 0], strides = [1, 1, 1, 1]}

# ==== BroadcastTo ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [1, 1, 1, 1], StorageShape = [1, 1, 1, 1], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 9, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 6, 24, 24], StorageShape = [4, 6, 24, 24], shapeRange = [], memtype = 0, isConst = 0

# ==== ReduceSum ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 2, StorageFormat = 2, Shape = [4, 6, 576], StorageShape = [4, 6, 576], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 9, Format = 2, StorageFormat = 2, Shape = [2], StorageShape = [2], shapeRange = [], memtype = 1, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 2, StorageFormat = 2, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {keep_dims = False}

# ==== Slice ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 4, Shape = [6, 1, 5, 5], StorageShape = [25, 1, 16, 16], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 9, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# InputDesc[2]: [TensorDesc] DataType = 9, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 4, Shape = [6, 1, 5, 5], StorageShape = [25, 1, 16, 16], shapeRange = [], memtype = 0, isConst = 0

loss.backward()

# ==== Identity ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 4, Shape = [6, 1, 5, 5], StorageShape = [25, 1, 16, 16], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6, 1, 5, 5], StorageShape = [6, 1, 5, 5], shapeRange = [], memtype = 0, isConst = 0
print("conv.weight.grad=", conv_2d.weight.grad)

print("conv.bias.grad=", conv_2d.bias.grad)

# ==== Identity ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [4, 1, 28, 28], StorageShape = [4, 1, 28, 28, 16], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 1, 28, 28], StorageShape = [4, 1, 28, 28], shapeRange = [], memtype = 0, isConst = 0
print("x_tensor.grad=", x_tensor.grad)
 
# typedef enum {
#     ACL_FORMAT_UNDEFINED = -1,
#     ACL_FORMAT_NCHW = 0,
#     ACL_FORMAT_NHWC = 1,
#     ACL_FORMAT_ND = 2,
#     ACL_FORMAT_NC1HWC0 = 3,
#     ACL_FORMAT_FRACTAL_Z = 4,
#     ACL_FORMAT_NC1HWC0_C04 = 12,
#     ACL_FORMAT_HWCN = 16,
#     ACL_FORMAT_NDHWC = 27,
#     ACL_FORMAT_FRACTAL_NZ = 29,
#     ACL_FORMAT_NCDHW = 30,
#     ACL_FORMAT_NDC1HWC0 = 32,
#     ACL_FRACTAL_Z_3D = 33
# } aclFormat;