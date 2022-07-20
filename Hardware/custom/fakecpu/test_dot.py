import paddle

# export FLAGS_enable_eager_mode=1
# paddle.fluid.framework._disable_legacy_dygraph()

print(f"EAGER_MODE={paddle.fluid.framework.in_dygraph_mode()}")

# paddle.set_device('cpu')
paddle.set_device('custom_cpu')

x = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32', stop_gradient=False)
y = paddle.to_tensor([[4, 3], [2, 1]], dtype='float32')

grad = paddle.to_tensor([[1, 1], [1, 1]], dtype='float32')
z = paddle.dot(x, y)

print(f"x = {x}")
print(f"y = {y}")
print(f"z = {z}")
print(f"grad = {grad}")

paddle.autograd.backward([z], [grad], False)
print(f"x.grad={x.grad.numpy()}")
# print(f"x.grad={x.grad}")
