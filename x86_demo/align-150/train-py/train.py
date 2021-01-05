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

# Init Conv Param
# input_data = np.arange(1, input_size+1, dtype=np.float32).reshape((batch_size, input_channel, input_height, input_width)) 
input_data = np.ones([batch_size, input_channel, input_height, input_width]).astype('float32')
print("Input Data = [{}, {}, {}, {}]".format(batch_size, input_channel, input_height, input_width))
for bs in range(batch_size):
    for ic in range(input_channel):
        for ih in range(input_height):
            print("[ {} ]".format(" ".join([str(float_formatter(v)) for v in input_data[bs, ic, ih, :]])))
        print("")
    print("-----------------------")

filter_data = np.arange(1, filter_size+1, dtype=np.float32).reshape((output_channel, 1, kernel_h, kernel_w)) 
print("Filter Data = [{}, {}, {}, {}]".format(output_channel, 1, kernel_h, kernel_w))
for oc in range(output_channel):
    for kh in range(kernel_h):
        print("[ {} ]".format(" ".join(str(float_formatter(v)) for v in filter_data[oc,0,kh,:])))
    print("")
print("-----------------------")

bias_data = np.arange(1, output_channel+1, dtype=np.float32).reshape((output_channel,))
bias_data /= 10
print("Bias Data = {}".format(output_channel))
print("[ {} ]".format(" ".join(str(float_formatter(v)) for v in bias_data)))
print("")

batchnorm_data = np.ones([output_channel,]).astype('float32')
print("Batch Norm Data = {}".format(output_channel))
print("[ {} ]".format(" ".join(str(float_formatter(v)) for v in batchnorm_data)))
print("")


# Build the model
main_prog = fluid.Program()
startup_prog = fluid.Program()
with fluid.program_guard(main_prog, startup_prog):
    # input
    data = fluid.layers.data(name="img", shape=[batch_size, input_channel, input_height, input_width], append_batch_size=False)
    # conv
    conv_weight_attr = fluid.ParamAttr(name="conv_weight_1", initializer=fluid.initializer.NumpyArrayInitializer(filter_data))
    conv_bias_attr = fluid.ParamAttr(name="conv_bias_1", initializer=fluid.initializer.NumpyArrayInitializer(bias_data))
    conv = fluid.layers.conv2d(input=data, 
                               num_filters=output_channel, 
                               filter_size=kernel_h, 
                               stride=conv_stride, 
                               padding=conv_padding, 
                               dilation=conv_dilation, 
                               groups=groups, 
                               param_attr=conv_weight_attr, 
                               bias_attr=conv_bias_attr)
    # batch_norm
    bn_scale_attr = fluid.ParamAttr(name="norm_scale_1", initializer=fluid.initializer.NumpyArrayInitializer(batchnorm_data))
    bn_bias_attr = fluid.ParamAttr(name="norm_bias_2", initializer=fluid.initializer.NumpyArrayInitializer(batchnorm_data))
    norm = paddle.fluid.layers.batch_norm(input=conv,
                                          param_attr=bn_scale_attr, 
                                          bias_attr=bn_bias_attr)
    # relu
    relu = paddle.fluid.layers.relu(norm)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_prog)

# Save the inference model
fluid.io.save_inference_model(dirname=save_dirname, feeded_var_names=['img'],
                target_vars=[norm], executor=exe, main_program=main_prog,
                model_filename="__model__", params_filename="__params__")

[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=save_dirname, 
                executor=exe, model_filename="__model__", params_filename="__params__")

output_tensor = exe.run(inference_program,
                  feed={feed_target_names[0]: input_data},
                  fetch_list=fetch_targets,
                  return_numpy=True)
output_data = np.array(output_tensor).reshape((batch_size, output_channel, output_height, output_width)) 
print(output_data.shape)

print("Output Data = [{}, {}, {}, {}]".format(batch_size, output_channel, output_height, output_width))
for bs in range(batch_size):
    for ic in range(output_channel):
        for ih in range(output_height):
            print("[ {} ]".format(" ".join([str(float_formatter(v)) for v in output_data[bs, ic, ih, :]])))
        print("")
    print("-----------------------")