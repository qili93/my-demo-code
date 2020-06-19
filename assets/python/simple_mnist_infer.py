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
import numpy
import paddle
import paddle.fluid as fluid

def infer_mnist(save_dirname):
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(3) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
    exe = fluid.Executor(place)

    def load_image(file):
        im = Image.open(file).convert('L')
        im = im.resize((28, 28), Image.ANTIALIAS)
        im = numpy.array(im).reshape(1, 1, 28, 28).astype(numpy.float32)
        im = im / 255.0 * 2.0 - 1.0
        return im

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(cur_dir, '../images/infer_3.png')
    tensor_img = load_image(img_path)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)
        results = exe.run(inference_program, feed={feed_target_names[0]: tensor_img}, fetch_list=fetch_targets)
        lab = numpy.argsort(results)
        print("Inference result of ../images/infer_3.png is: %d" % lab[0][0][-1])

if __name__ == '__main__':
    infer_mnist(save_dirname='../models/simple_mnist')