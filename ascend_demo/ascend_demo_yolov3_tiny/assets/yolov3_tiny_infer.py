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

import os
from PIL import Image
import numpy as np
import paddle
import paddle.fluid as fluid

def infer_yolov3(save_dirname):
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()

    # prepare inputs
    input_data_low = np.full((BATCH_SIZE, 21, 6, 6), 1).astype('float32')
    input_data_mid = np.full((BATCH_SIZE, 21, 12, 12), 1).astype('float32')
    input_data_high = np.full((BATCH_SIZE, 21, 24, 24), 1).astype('float32')
    input_imgsize = np.full((BATCH_SIZE, 2), 1).astype('int32')

    # inference
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)
        # for var_name in feed_target_names:
        #   print("feed_target_name is ", var_name)
        with fluid.program_guard(inference_program, fluid.Program()):
          results = exe.run(inference_program, 
                            feed={
                              feed_target_names[0]: input_data_low, 
                              feed_target_names[1]: input_data_mid, 
                              feed_target_names[2]: input_data_high, 
                              feed_target_names[3]: input_imgsize }, 
                            fetch_list=fetch_targets,
                            return_numpy=False)
        print(np.array(results[0]).shape)
        print(np.array(results[0]))

if __name__ == '__main__':
    BATCH_SIZE = 1
    infer_yolov3(save_dirname='./models/yolov3_tiny_model')