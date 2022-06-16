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
import paddleslim as slim
import numpy as np


# Load the inference model
place = fluid.CPUPlace()
exe = fluid.Executor(place)

path_infer = './mnist_infer'
path_quant = './mnist_quant'

def batch_generator_creator():
        def __reader__():
            for _ in range(16):
                tensor_img = np.array(np.random.random((1, 3, 4, 4)), dtype=np.float32)
                yield tensor_img
    
        return __reader__

# tensor_img = np.array(np.random.random((1, 3, 4, 4)), dtype=np.float32)
slim.quant.quant_post_static(
        executor=exe,
        model_dir=path_infer,
        quantize_model_path=path_quant,
        sample_generator=batch_generator_creator(),
        batch_nums=1)
