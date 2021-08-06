from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np

paddle.enable_static()

# Build the model
main_prog = fluid.Program()
startup_prog = fluid.Program()
with fluid.program_guard(main_prog, startup_prog):
    data = fluid.layers.data(name="img", shape=[1, 3, 4, 4], append_batch_size=False)
    conv = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, stride=1)
    # [1, 2(num_filters), 2, 2] # 2 = 4 - filter_size + 1 when stride=1
    norm = paddle.fluid.layers.batch_norm(input=conv)
    relu = paddle.fluid.layers.relu(x=norm)
    # [1, 2, 2, 2]
    fc = paddle.fluid.layers.fc(input=relu, size=2)
    # [1, 2]
    soft = fluid.layers.softmax(input=fc, axis=1)
    bias = fluid.layers.create_parameter(shape=[1], dtype='float32')
    out = fluid.layers.elementwise_add(soft, bias)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_prog)

# Save the inference model
path = './custom_model'
fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'],
                target_vars=[out], executor=exe, main_program=main_prog)

[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)
# tensor_img = np.array([[[[1.0], [2.0], [3.0]],[[4.0], [5.0], [6.0]]]], 'float32') # [1, 2, 3, 1]
tensor_img = np.full((1, 3, 4, 4), 1, 'float32')
results = exe.run(inference_program,
                  feed={feed_target_names[0]: tensor_img},
                  fetch_list=fetch_targets)
print(results)