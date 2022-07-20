import paddle
import numpy as np

print(f"Legacy Mode = {paddle.fluid.framework._in_legacy_dygraph()}")

paddle.set_device('cpu')
# paddle.set_device('custom_cpu')

x = np.random.random([2, 2]).astype("float32")
y = np.random.random([2, 2]).astype("float32")
grad = np.ones([2, 2]).astype("float32")
x_grad = 2 * np.matmul(grad, y.transpose())
print(f"x_grad={x_grad}")


x_tensor = paddle.to_tensor(x, stop_gradient=False)
y_tensor = paddle.to_tensor(y)
z1_tensor = paddle.matmul(x_tensor, y_tensor)
z2_tensor = paddle.matmul(x_tensor, y_tensor)

grad_tensor = paddle.to_tensor(grad)
paddle.autograd.backward([z1_tensor, z2_tensor], [grad_tensor, None])
print(f"x_tensor.grad.numpy()={x_tensor.grad.numpy()}")


# x = np.random.random([2, 2]).astype("float32")
# y = np.random.random([2, 2]).astype("float32")
# grad = np.ones([2, 2]).astype("float32")

# x_tensor = paddle.to_tensor(x, stop_gradient=False)
# y_tensor = paddle.to_tensor(y)

# z1_tensor = paddle.matmul(x_tensor, y_tensor)
# z2_tensor = paddle.matmul(x_tensor, y_tensor)

# grad_tensor = paddle.to_tensor(grad)

# paddle.autograd.backward([z1_tensor, z2_tensor], [grad_tensor, None])

# print(f"x_tensor.grad={x_tensor.grad.numpy()}")

# print("=====================")

# print(f"x_grad={2 * np.matmul(grad, y.transpose())}")
