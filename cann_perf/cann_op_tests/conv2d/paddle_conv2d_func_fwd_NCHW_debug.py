import os
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

os.environ['FLAGS_npu_storage_format'] = "True"

paddle.set_device("npu")

# input = paddle.ones(shape=[4, 1, 28, 28])
# filter = paddle.ones(shape=[6, 1, 5, 5])
# bias = paddle.ones(shape=[6])

input = paddle.ones(shape=[2, 1, 4, 4])
# print(f"input={input}")
# input=Tensor(shape=[2, 1, 4, 4], dtype=float32, place=Place(npu:0), stop_gradient=True,
#        [[[[1., 1., 1., 1.],
#           [1., 1., 1., 1.],
#           [1., 1., 1., 1.],
#           [1., 1., 1., 1.]]],


#         [[[1., 1., 1., 1.],
#           [1., 1., 1., 1.],
#           [1., 1., 1., 1.],
#           [1., 1., 1., 1.]]]])

filter = paddle.ones(shape=[3, 1, 3, 3])
# print(f"filter={filter}")
# filter=Tensor(shape=[3, 1, 3, 3], dtype=float32, place=Place(npu:0), stop_gradient=True,
#        [[[[1., 1., 1.],
#           [1., 1., 1.],
#           [1., 1., 1.]]],


#         [[[1., 1., 1.],
#           [1., 1., 1.],
#           [1., 1., 1.]]],


#         [[[1., 1., 1.],
#           [1., 1., 1.],
#           [1., 1., 1.]]]])


bias = paddle.ones(shape=[3])
# print(f"bias={bias}")
# bias=Tensor(shape=[3], dtype=float32, place=Place(npu:0), stop_gradient=True,
#        [1., 1., 1.])

# ==== Conv2D ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [2, 1, 4, 4], StorageShape = [2, 1, 4, 4], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [3, 1, 3, 3], StorageShape = [3, 1, 3, 3], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [2, 3, 4, 4], StorageShape = [2, 1, 4, 4, 16], shapeRange = [], memtype = 0, isConst = 0
# Attr: {data_format = NCHW, dilations = [1, 1, 1, 1], groups = 1, pads = [1, 1, 1, 1], strides = [1, 1, 1, 1]}

# ==== Add ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [2, 3, 4, 4], StorageShape = [2, 1, 4, 4, 16], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [1, 3, 1, 1], StorageShape = [1, 1, 1, 1, 16], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [2, 3, 4, 4], StorageShape = [2, 1, 4, 4, 16], shapeRange = [], memtype = 0, isConst = 0

out = F.conv2d(input, filter, bias, padding=1)

# ==== Identity ====
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [2, 3, 4, 4], StorageShape = [2, 1, 4, 4, 16], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [2, 3, 4, 4], StorageShape = [2, 3, 4, 4], shapeRange = [], memtype = 0, isConst = 0

# print(f"out={out}")

# print(f"out.numpy(0)={out.numpy()}")

# -----

# out_tensor = out.value().get_tensor()

# print(f"out_tensor={out_tensor}") # NPU:0 TENSOR

print(f"out.numpy(1)={np.array(out.value().get_tensor())}") # MEMCOPY TO CPU

# out=Tensor(shape=[2, 3, 4, 4], dtype=float32, place=Place(npu:0), stop_gradient=True,
#        [[[[5. , 7. , 7. , 5. ],
#           [7. , 10., 10., 7. ],
#           [7. , 10., 10., 7. ],
#           [5. , 7. , 7. , 5. ]],

#          [[5. , 7. , 7. , 5. ],
#           [7. , 10., 10., 7. ],
#           [7. , 10., 10., 7. ],
#           [5. , 7. , 7. , 5. ]],

#          [[5. , 7. , 7. , 5. ],
#           [7. , 10., 10., 7. ],
#           [7. , 10., 10., 7. ],
#           [5. , 7. , 7. , 5. ]]],


#         [[[5. , 7. , 7. , 5. ],
#           [7. , 10., 10., 7. ],
#           [7. , 10., 10., 7. ],
#           [5. , 7. , 7. , 5. ]],

#          [[5. , 7. , 7. , 5. ],
#           [7. , 10., 10., 7. ],
#           [7. , 10., 10., 7. ],
#           [5. , 7. , 7. , 5. ]],

#          [[5. , 7. , 7. , 5. ],
#           [7. , 10., 10., 7. ],
#           [7. , 10., 10., 7. ],
#           [5. , 7. , 7. , 5. ]]]])

