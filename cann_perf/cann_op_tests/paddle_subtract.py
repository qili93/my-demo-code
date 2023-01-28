import paddle

paddle.set_device("npu:0")
# paddle.set_device("cpu")

x = paddle.to_tensor([[1, 2], [7, 8]], dtype="float32")
y = paddle.to_tensor([[5, 6], [3, 4]], dtype="float32")
x.stop_gradient = False
y.stop_gradient = False

res = paddle.subtract(x, y)

loss = res.sum()
loss.backward()

print("x.grad=", x.grad)
print("y.grad=", y.grad)
