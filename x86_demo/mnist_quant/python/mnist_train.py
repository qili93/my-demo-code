#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    relu = fluid.layers.relu(data)
    conv = fluid.layers.conv2d(relu, 2, 3)
    # [1, 2, 2, 2]
    flat = fluid.layers.flatten(conv, 1)
    # [1, 8]
    bias = fluid.layers.create_parameter(shape=[8], dtype='float32')
    out = fluid.layers.elementwise_add(flat, bias)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_prog)

# Save the inference model
path = './mnist_quant'
# path = '../assets/models/mnist_quant'
fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'],
                target_vars=[out], executor=exe, main_program=main_prog)

[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)
tensor_img = np.array(np.random.random((1, 3, 4, 4)), dtype=np.float32)
results = exe.run(inference_program,
                  feed={feed_target_names[0]: tensor_img},
                  fetch_list=fetch_targets)