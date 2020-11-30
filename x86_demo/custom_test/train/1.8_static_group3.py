from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np

# format
float_formatter = "{:9.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# Build the model
main_prog = fluid.Program()
startup_prog = fluid.Program()
with fluid.program_guard(main_prog, startup_prog):
    data = fluid.layers.data(name="img", shape=[1, 3, 2, 2], append_batch_size=False)
    weight_data_1 = np.array([0.01, 0.1, 1, 10, 100, 1000]).astype(np.float32).reshape([6, 1, 1, 1])
    weight_attr_1 = fluid.ParamAttr(name="conv_weight_1", initializer=fluid.initializer.NumpyArrayInitializer(weight_data_1))
    # bias_data_1 = np.array([1, 2]).astype(np.float32).reshape([6])
    bias_data_1 = np.arange(1,7).astype(np.float32).reshape([6])
    bias_attr_1 = fluid.ParamAttr(name="conv_bias_1", initializer=fluid.initializer.NumpyArrayInitializer(bias_data_1))
    conv = fluid.layers.conv2d(input=data, 
                               num_filters=6, 
                               filter_size=1, 
                               stride=1, 
                               padding=0, 
                               dilation=1, 
                               groups=3, 
                               param_attr=weight_attr_1, 
                               bias_attr=bias_attr_1)
    # [1, 2(num_filters), 2, 2] # 2 = 4 - filter_size + 1 when stride=1
    scale_data_1 = np.array([1000, 100, 10, 1, 0.1, 0.01]).astype(np.float32).reshape([6])
    scale_attr_1 = fluid.ParamAttr(name="norm_scale_1", initializer=fluid.initializer.NumpyArrayInitializer(scale_data_1))
    # bias_data_2 = np.array([1, 2]).astype(np.float32).reshape([6])
    # bias_data_2 = np.arange(1,7).astype(np.float32).reshape([6])
    bias_data_2 = np.full((6), 0.01).astype(np.float32).reshape([6])
    bias_attr_2 = fluid.ParamAttr(name="norm_bias_2", initializer=fluid.initializer.NumpyArrayInitializer(bias_data_2))
    norm = paddle.fluid.layers.batch_norm(input=conv,
                                          param_attr=scale_attr_1, 
                                          bias_attr=bias_attr_2)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_prog)

# Save the inference model
path = './model_group3'
fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'],
                target_vars=[norm], executor=exe, main_program=main_prog)

[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)

input_data = np.arange(1, 13).astype(np.float32).reshape([1, 3, 2, 2])
print("input shape is {}".format(input_data.shape))
print("input data is \n {}".format(input_data))
# tensor_img = np.array([[[[1.0], [2.0], [3.0]],[[4.0], [5.0], [6.0]]]], 'float32') # [1, 2, 3, 1]
# tensor_img = np.full((1, 3, 4, 4), 1, 'float32')
results = exe.run(inference_program,
                  feed={feed_target_names[0]: input_data},
                  fetch_list=fetch_targets)
print(results)