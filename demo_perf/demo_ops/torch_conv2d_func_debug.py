import torch
import torch.npu
import torch.nn.functional as F


# input = torch.ones(4, 1, 28, 28).to("npu:0")
# filter = torch.ones(6, 1, 5, 5).to("npu:0")
# bias = torch.ones(6).to("npu:0")

input = torch.ones(2, 1, 4, 4).to("npu:0")
print(f"input={input}")
# input=tensor([[[[1., 1., 1., 1.],
#           [1., 1., 1., 1.],
#           [1., 1., 1., 1.],
#           [1., 1., 1., 1.]]],


#         [[[1., 1., 1., 1.],
#           [1., 1., 1., 1.],
#           [1., 1., 1., 1.],
#           [1., 1., 1., 1.]]]], device='npu:0')

filter = torch.ones(3, 1, 3, 3).to("npu:0")
print(f"filter={filter}")
# filter=tensor([[[[1., 1., 1.],
#           [1., 1., 1.],
#           [1., 1., 1.]]],


#         [[[1., 1., 1.],
#           [1., 1., 1.],
#           [1., 1., 1.]]],


#         [[[1., 1., 1.],
#           [1., 1., 1.],
#           [1., 1., 1.]]]], device='npu:0')

bias = torch.ones(3).to("npu:0")
print(f"bias={bias}")
# bias=tensor([1., 1., 1.], device='npu:0')


# ==== Conv2D ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [2, 1, 4, 4], StorageShape = [2, 1, 4, 4], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [3, 1, 3, 3], StorageShape = [3, 1, 3, 3], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[2]: [TensorDesc] DataType = 0, Format = 2, StorageFormat = 2, Shape = [3], StorageShape = [3], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [2, 3, 4, 4], StorageShape = [2, 1, 4, 4, 16], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {data_format = NCHW, dilations = [1, 1, 1, 1], groups = 1, pads = [1, 1, 1, 1], strides = [1, 1, 1, 1]}
out = F.conv2d(input, filter, bias, padding=1)

# ==== Identity ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [2, 3, 4, 4], StorageShape = [2, 1, 4, 4, 16], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [2, 3, 4, 4], StorageShape = [2, 3, 4, 4], shapeRange = [], memtype = 0, isConst = 0
print(f"out={out}")
# out=tensor([[[[ 5.,  7.,  7.,  5.],
#           [ 7., 10., 10.,  7.],
#           [ 7., 10., 10.,  7.],
#           [ 5.,  7.,  7.,  5.]],

#          [[ 5.,  7.,  7.,  5.],
#           [ 7., 10., 10.,  7.],
#           [ 7., 10., 10.,  7.],
#           [ 5.,  7.,  7.,  5.]],

#          [[ 5.,  7.,  7.,  5.],
#           [ 7., 10., 10.,  7.],
#           [ 7., 10., 10.,  7.],
#           [ 5.,  7.,  7.,  5.]]],


#         [[[ 5.,  7.,  7.,  5.],
#           [ 7., 10., 10.,  7.],
#           [ 7., 10., 10.,  7.],
#           [ 5.,  7.,  7.,  5.]],

#          [[ 5.,  7.,  7.,  5.],
#           [ 7., 10., 10.,  7.],
#           [ 7., 10., 10.,  7.],
#           [ 5.,  7.,  7.,  5.]],

#          [[ 5.,  7.,  7.,  5.],
#           [ 7., 10., 10.,  7.],
#           [ 7., 10., 10.,  7.],
#           [ 5.,  7.,  7.,  5.]]]], device='npu:0')
