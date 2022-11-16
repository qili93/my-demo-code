import torch
import torch.npu
import torch.nn.functional as F


input = torch.ones(4, 1, 28, 28).to("npu:0")
filter = torch.ones(6, 1, 5, 5).to("npu:0")
bias = torch.ones(6).to("npu:0")
input.requires_grad = True
filter.requires_grad = True
bias.requires_grad = True


# opType = Conv2D
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 1, 28, 28], StorageShape = [4, 1, 28, 28], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6, 1, 5, 5], StorageShape = [6, 1, 5, 5], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[2]: [TensorDesc] DataType = 0, Format = 2, StorageFormat = 2, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [4, 6, 26, 26], StorageShape = [4, 1, 26, 26, 16], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {data_format = NCHW, dilations = [1, 1, 1, 1], groups = 1, pads = [1, 1, 1, 1], strides = [1, 1, 1, 1]}
out = F.conv2d(input, filter, bias, padding=1)

# opType = ReduceSum
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [4, 6, 26, 26], StorageShape = [4, 1, 26, 26, 16], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 9, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 2, StorageFormat = 2, Shape = [], StorageShape = [], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {keep_dims = False}
loss = out.sum()


# opType = OnesLike
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 2, StorageFormat = -1, Shape = [], StorageShape = [], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 2, StorageFormat = 2, Shape = [], StorageShape = [], shapeRange = [], memtype = 0, isConst = 0

# opType = BroadcastTo
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [1, 1, 1, 1], StorageShape = [1, 1, 1, 1], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 9, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 6, 26, 26], StorageShape = [4, 6, 26, 26], shapeRange = [], memtype = 0, isConst = 0

# opType = Conv2DBackpropInput
# InputDesc[0]: [TensorDesc] DataType = 3, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6, 1, 5, 5], StorageShape = [6, 1, 5, 5], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[2]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 6, 26, 26], StorageShape = [4, 6, 26, 26], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [4, 1, 28, 28], StorageShape = [4, 1, 28, 28, 16], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {data_format = NCHW, dilations = [1, 1, 1, 1], groups = 1, pads = [1, 1, 1, 1], strides = [1, 1, 1, 1]}

# opType = BroadcastTo
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [1, 1, 1, 1], StorageShape = [1, 1, 1, 1], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 9, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 6, 26, 26], StorageShape = [4, 6, 26, 26], shapeRange = [], memtype = 0, isConst = 0

# opType = Conv2DBackpropFilter
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 1, 28, 28], StorageShape = [4, 1, 28, 28], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 3, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# InputDesc[2]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 6, 26, 26], StorageShape = [4, 6, 26, 26], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 4, Shape = [6, 1, 5, 5], StorageShape = [25, 1, 16, 16], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {data_format = NCHW, dilations = [1, 1, 1, 1], groups = 1, pads = [1, 1, 1, 1], strides = [1, 1, 1, 1]}

# opType = BroadcastTo
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [1, 1, 1, 1], StorageShape = [1, 1, 1, 1], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 9, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 6, 26, 26], StorageShape = [4, 6, 26, 26], shapeRange = [], memtype = 0, isConst = 0

# opType = ReduceSum
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 2, StorageFormat = 2, Shape = [4, 6, 676], StorageShape = [4, 6, 676], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 9, Format = 2, StorageFormat = 2, Shape = [2], StorageShape = [2], shapeRange = [], memtype = 1, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 2, StorageFormat = 2, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {keep_dims = False}

# opType = Slice
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 4, Shape = [6, 1, 5, 5], StorageShape = [25, 1, 16, 16], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 9, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# InputDesc[2]: [TensorDesc] DataType = 9, Format = 2, StorageFormat = 2, Shape = [4], StorageShape = [4], shapeRange = [], memtype = 1, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 4, Shape = [6, 1, 5, 5], StorageShape = [25, 1, 16, 16], shapeRange = [], memtype = 0, isConst = 0

loss.backward()

# opType = Identity
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 4, Shape = [6, 1, 5, 5], StorageShape = [25, 1, 16, 16], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6, 1, 5, 5], StorageShape = [6, 1, 5, 5], shapeRange = [], memtype = 0, isConst = 0
print("filter.grad=", filter.grad)


print("bias.grad=", bias.grad)


# opType = Identity
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [4, 1, 28, 28], StorageShape = [4, 1, 28, 28, 16], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 1, 28, 28], StorageShape = [4, 1, 28, 28], shapeRange = [], memtype = 0, isConst = 0
print("input.grad=", input.grad)
