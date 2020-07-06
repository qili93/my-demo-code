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
import shutil
import pathlib
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import TracedLayer

class RELU(fluid.dygraph.Layer):
    def __init__(self):
        super(RELU, self).__init__()

    def forward(self, inputs, label=None):
        # [-1, 1, 28, 28]
        output = fluid.layers.relu(inputs)
        return output

def save_model(save_dirname):
    place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()

    with fluid.dygraph.guard(place):
        model = RELU()
        # save inference model
        if save_dirname is None:
            return
        # delete old model
        if  os.path.exists(save_dirname):
            shutil.rmtree(save_dirname)
            os.makedirs(save_dirname)
        # save inference model
        in_np = np.random.random([1, 1, 28, 28]).astype('float32')
        input_var = fluid.dygraph.to_variable(in_np)
        out_dygraph, static_layer = TracedLayer.trace(model, inputs=[input_var])
        
        static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])
        print("Saved inference model to {}".format(save_dirname))

if __name__ == '__main__':
    save_model(save_dirname='../models/mnist_model')