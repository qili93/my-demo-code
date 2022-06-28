import paddle

# paddle.set_device('cpu')
paddle.set_device('custom_cpu')

x = paddle.to_tensor([[1, 2], [3, 4]], dtype='float64', stop_gradient=False)
y = paddle.to_tensor([[3, 2], [3, 4]], dtype='float64')

grad_tensor1 = paddle.to_tensor([[1,2], [2, 3]], dtype='float64')
grad_tensor2 = paddle.to_tensor([[1,1], [1, 1]], dtype='float64')

z1 = paddle.add(x, y)
z2 = paddle.add(x, y)

paddle.autograd.backward([z1, z2], [grad_tensor1, grad_tensor2], True)
print(f"x.grad={x.grad.numpy()}")
# print(f"x.grad={x.grad}") # will convert tensor data to fp32

