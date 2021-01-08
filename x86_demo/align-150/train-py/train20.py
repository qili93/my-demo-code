from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np

# format
float_formatter = "{:9.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

save_dirname = './dconv08'

# Define Conv Attr
# input
batch_size = 1
input_channel = 8
input_height = 64
input_width = 64
input_size = batch_size * input_channel * input_height * input_width
# filter
output_channel = 8
groups = 8
kernel_h = 3
kernel_w = 3
filter_size = output_channel * 1 * kernel_h * kernel_w
# attr
conv_stride = 1
conv_padding = 1
conv_dilation = 1
# output
output_height = input_height
output_width = input_width


input_data = np.ones([batch_size, input_channel, input_height, input_width]).astype('float32')
conv_filter_data =  np.load('data/conv_filter.npy')
conv_bias_data = np.zeros([output_channel]).astype('float32')
bn_bias_data =  np.load('data/bn_bias.npy')
bn_scale_data =  np.load('data/bn_scale.npy')
bn_mean_data =  np.load('data/bn_mean.npy')
bn_var_data =  np.load('data/bn_var.npy')

class ExampleLayer(paddle.nn.Layer):
    def __init__(self):
        super(ExampleLayer, self).__init__()
        conv_weight_attr = paddle.framework.ParamAttr(name="conv_weight_1", initializer=paddle.nn.initializer.Assign(conv_filter_data))
        conv_bias_attr = paddle.framework.ParamAttr(name="conv_bias_1", initializer=paddle.nn.initializer.Assign(conv_bias_data))
        self._conv2d = paddle.nn.Conv2D(in_channels=input_channel, 
                                        out_channels=output_channel, 
                                        kernel_size=kernel_h, 
                                        groups=groups, 
                                        stride=conv_stride,
                                        padding=conv_padding,
                                        weight_attr=conv_weight_attr,
                                        bias_attr=conv_bias_attr)
        self._relu = paddle.nn.ReLU()

    def forward(self, input):
        x = self._conv2d(input)
        rm = paddle.to_tensor(bn_mean_data)
        rv = paddle.to_tensor(bn_var_data)
        w = paddle.to_tensor(bn_scale_data)
        b = paddle.to_tensor(bn_bias_data)
        x = paddle.nn.functional.batch_norm(x, rm, rv, w, b, data_format="NCHW")
        x = self._relu(x)
        return x

input_tensor = paddle.to_tensor(input_data)
layer = ExampleLayer()

# Save the inference model
out_dygraph, static_layer = paddle.jit.TracedLayer.trace(layer, inputs=[input_tensor])
static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])