import paddle
import numpy as np

class ExampleLayer(paddle.nn.Layer):
    def __init__(self):
        super(ExampleLayer, self).__init__()
        # weight_data_1 = np.arange(1,7).astype(np.float32).reshape([6, 1, 1, 1])
        weight_data_1 = np.array([0.01, 0.1, 1, 10, 100, 1000]).astype(np.float32).reshape([6, 1, 1, 1])
        weight_attr_1 = paddle.framework.ParamAttr(name="conv_weight_1",
                        initializer=paddle.nn.initializer.Assign(weight_data_1))
        bias_data_1 = np.full((6), 1).astype(np.float32).reshape([6])
        # bias_data_1 = np.arange(1,7).astype(np.float32).reshape([6])
        bias_attr_1 = paddle.framework.ParamAttr(name="conv_bias_1",
                      initializer=paddle.nn.initializer.Assign(bias_data_1))
        self._conv2d = paddle.nn.Conv2D(in_channels=3, 
                                        out_channels=6, 
                                        kernel_size=1, 
                                        groups=3, 
                                        stride=1,
                                        padding=0,
                                        weight_attr=weight_attr_1,
                                        bias_attr=bias_attr_1)

    def forward(self, input):
        return self._conv2d(input)

save_dirname = './saved_infer_model'

# input_data = np.array([1, 10, 100]).astype(np.float32).reshape([1, 3, 1, 1])
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
input_data = np.arange(1, 13).astype(np.float32).reshape([1, 3, 2, 2])
print("input shape is {}".format(input_data.shape)) # (1, 3, 2, 2)
print("input data is \n {}".format(input_data))
input_tensor = paddle.to_tensor(input_data)
layer = ExampleLayer()

out_dygraph, static_layer = paddle.jit.TracedLayer.trace(layer, inputs=[input_tensor])
static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])

paddle.enable_static()
place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
program, feed_vars, fetch_vars = paddle.static.load_inference_model(save_dirname, exe)

fetch, = exe.run(program, feed={feed_vars[0]: input_data}, fetch_list=fetch_vars)
print("output shape is {}".format(fetch.shape)) # (1, 6, 1, 1)
print("output data is \n {}".format(fetch)) 
# NO BIAS =>> (1, 2, 30, 40, 500, 600)
# BIAS DATA = (1, 2,  3,  4,   5,   6)
# BIAS OUT => (2, 4, 33, 44, 505, 606)