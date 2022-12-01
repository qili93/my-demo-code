import paddle

paddle.set_device("npu")
linear = paddle.nn.Linear(3, 3)
print(linear.weight) # uniform_random
print(linear.bias) # full_
