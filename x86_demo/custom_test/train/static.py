import numpy as np
import paddle

class ExampleLayer(paddle.nn.Layer):
    def __init__(self):
        super(ExampleLayer, self).__init__()
        self._conv2d = paddle.nn.Conv2D(in_channels=3, out_channels=2, kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        return self._conv2d(input)

save_dirname = './saved_infer_model'
in_np = np.random.random([1, 3, 4, 4]).astype('float32')
in_var = paddle.to_tensor(in_np)
layer = ExampleLayer()

out_dygraph, static_layer = paddle.jit.TracedLayer.trace(layer, inputs=[in_var])
static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])

paddle.enable_static()
place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
program, feed_vars, fetch_vars = paddle.static.load_inference_model(save_dirname, exe)

fetch, = exe.run(program, feed={feed_vars[0]: in_np}, fetch_list=fetch_vars)
print(fetch.shape) # (1, 8, 10)