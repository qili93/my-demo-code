import os
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

os.environ['FLAGS_npu_storage_format'] = "1"

paddle.set_device("npu")

input = paddle.ones(shape=[2, 4, 4, 1])
# print(f"input={input}")
# input=Tensor(shape=[2, 4, 4, 1], dtype=float32, place=Place(npu:0), stop_gradient=True,
#        [[[[1.],
#           [1.],
#           [1.],
#           [1.]],

#          [[1.],
#           [1.],
#           [1.],
#           [1.]],

#          [[1.],
#           [1.],
#           [1.],
#           [1.]],

#          [[1.],
#           [1.],
#           [1.],
#           [1.]]],


#         [[[1.],
#           [1.],
#           [1.],
#           [1.]],

#          [[1.],
#           [1.],
#           [1.],
#           [1.]],

#          [[1.],
#           [1.],
#           [1.],
#           [1.]],

#          [[1.],
#           [1.],
#           [1.],
#           [1.]]]])

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


out = F.conv2d(input, filter, bias, padding=1, data_format='NHWC')


print(f"out={out}")
# out=Tensor(shape=[2, 4, 4, 3], dtype=float32, place=Place(npu:0), stop_gradient=True,
#        [[[[5. , 5. , 5. ],
#           [7. , 7. , 7. ],
#           [7. , 7. , 7. ],
#           [5. , 5. , 5. ]],

#          [[7. , 7. , 7. ],
#           [10., 10., 10.],
#           [10., 10., 10.],
#           [7. , 7. , 7. ]],

#          [[7. , 7. , 7. ],
#           [10., 10., 10.],
#           [10., 10., 10.],
#           [7. , 7. , 7. ]],

#          [[5. , 5. , 5. ],
#           [7. , 7. , 7. ],
#           [7. , 7. , 7. ],
#           [5. , 5. , 5. ]]],


#         [[[5. , 5. , 5. ],
#           [7. , 7. , 7. ],
#           [7. , 7. , 7. ],
#           [5. , 5. , 5. ]],

#          [[7. , 7. , 7. ],
#           [10., 10., 10.],
#           [10., 10., 10.],
#           [7. , 7. , 7. ]],

#          [[7. , 7. , 7. ],
#           [10., 10., 10.],
#           [10., 10., 10.],
#           [7. , 7. , 7. ]],

#          [[5. , 5. , 5. ],
#           [7. , 7. , 7. ],
#           [7. , 7. , 7. ],
#           [5. , 5. , 5. ]]]])