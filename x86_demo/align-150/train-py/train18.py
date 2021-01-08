from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np

# for paddle 2.0
paddle.enable_static()

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
conv_filter =  np.load('data/conv_filter.npy')
conv_bias = np.zeros([output_channel]).astype('float32')
bn_bias =  np.load('data/bn_bias.npy')
bn_scale =  np.load('data/bn_scale.npy')
bn_mean =  np.load('data/bn_mean.npy')
bn_var =  np.load('data/bn_var.npy')

# Build the model
main_prog = fluid.Program()
startup_prog = fluid.Program()
with fluid.program_guard(main_prog, startup_prog):
    # input
    data = fluid.layers.data(name="img", shape=[batch_size, input_channel, input_height, input_width], append_batch_size=False)
    # conv
    conv_weight_attr = fluid.ParamAttr(name="conv_weight_1", initializer=fluid.initializer.NumpyArrayInitializer(conv_filter))
    conv_bias_attr = fluid.ParamAttr(name="conv_bias_1", initializer=fluid.initializer.NumpyArrayInitializer(conv_bias))
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
    bn_scale_attr = fluid.ParamAttr(name="norm_scale_1", initializer=fluid.initializer.NumpyArrayInitializer(bn_scale))
    bn_bias_attr = fluid.ParamAttr(name="norm_bias_1", initializer=fluid.initializer.NumpyArrayInitializer(bn_bias))
    bn_mean_attr = fluid.layers.create_global_var(name="norm_mean_1", shape=[8], value=1.0, dtype='float32', persistable=True)
    bn_var_attr = fluid.layers.create_global_var(name="norm_var_1", shape=[8], value=1.0, dtype='float32', persistable=True)

    fluid.layers.assign(bn_mean, bn_mean_attr)
    fluid.layers.assign(bn_var, bn_var_attr)
    norm = paddle.fluid.layers.batch_norm(input=conv,
                                          param_attr=bn_scale_attr, 
                                          bias_attr=bn_bias_attr,
                                          moving_mean_name="norm_mean_1",
                                          moving_variance_name="norm_var_1",
                                          is_test=True)
    # relu
    relu = paddle.fluid.layers.relu(norm)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_prog)

# Save the inference model
fluid.io.save_inference_model(dirname=save_dirname, feeded_var_names=['img'],
                target_vars=[norm], executor=exe, main_program=main_prog)

[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=save_dirname, 
                executor=exe)

output_tensor = exe.run(inference_program,
                  feed={feed_target_names[0]: input_data},
                  fetch_list=fetch_targets,
                  return_numpy=True)
output_data = np.array(output_tensor).reshape((batch_size, output_channel, output_height, output_width)) 

print("output_data shape is : "+str(output_data.shape))
# save to txt file
np.savetxt("infer-out-0.txt", output_data.flatten(), fmt='%10.3f')