import paddle
import paddle.nn as nn

paddle.set_device("cpu")
# paddle.set_device("ascend")

batch_norm_2d = nn.BatchNorm2D(6)

for name, param in batch_norm_2d.named_parameters():
    print(name, param)      # will print w_tmp,_linear.weight,_linear.bias

# weight Parameter containing:
# Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=False,
#        [1., 1., 1., 1., 1., 1.])
# bias Parameter containing:
# Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=False,
#        [0., 0., 0., 0., 0., 0.])
# _mean Parameter containing:
# Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [0., 0., 0., 0., 0., 0.])
# _variance Parameter containing:
# Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [1., 1., 1., 1., 1., 1.])
