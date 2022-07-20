import paddle

# export FLAGS_enable_eager_mode=1
paddle.fluid.framework._disable_legacy_dygraph()

#paddle.set_device('cpu')
paddle.set_device('custom_cpu')

x = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32', stop_gradient=False)
y = paddle.to_tensor([[3, 2], [3, 4]], dtype='float32')

grad_tensor = paddle.to_tensor([[1,2], [2, 3]], dtype='float32')
z = paddle.add(x, y)

print(f"x = {x}")
print(f"y = {y}")
print(f"z = {z}")
print(f"grad_tensor = {grad_tensor}")

# paddle.autograd.backward([z])

paddle.autograd.backward([z], [grad_tensor], False)
print(f"x.grad={x.grad.numpy()}")
# print(f"x.grad={x.grad}")
