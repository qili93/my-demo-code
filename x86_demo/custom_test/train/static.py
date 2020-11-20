import numpy as np
import paddle

class ExampleLayer(paddle.nn.Layer):
    def __init__(self):
        super(ExampleLayer, self).__init__()
        self._fc = paddle.nn.Linear(3, 10)

    def forward(self, input):
        return self._fc(input)

save_dirname = './saved_infer_model'
in_np = np.random.random([2, 3]).astype('float32')
in_var = paddle.to_tensor(in_np)
layer = ExampleLayer()

out_dygraph, static_layer = paddle.jit.TracedLayer.trace(layer, inputs=[in_var])
static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])

paddle.enable_static()
place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
program, feed_vars, fetch_vars = paddle.static.load_inference_model(save_dirname, exe)

fetch, = exe.run(program, feed={feed_vars[0]: in_np}, fetch_list=fetch_vars)
print(fetch.shape) # (2, 10)