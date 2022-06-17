#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from PIL import Image
from paddle.inference import Config
from paddle.inference import create_predictor

def infer_mnist(saved_model, image_path):
    if saved_model is None:
        return

    # create config
    config = Config(saved_model+'.pdmodel', saved_model+'.pdiparams')

    # enable custom device
    # config.enable_custom_device("custom_cpu")
    
    # create predictor
    predictor = create_predictor(config)

    # load image
    def load_image(file):
        im = Image.open(file).convert('L')
        im = im.resize((28, 28), Image.ANTIALIAS)
        im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
        im = im / 255.0 * 2.0 - 1.0
        return im
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(cur_dir, image_path)
    tensor_img = load_image(img_path)

    # set input
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])
    input_tensor.reshape(tensor_img.shape)
    input_tensor.copy_from_cpu(tensor_img.copy())

    # Run
    predictor.run()

    # Set output
    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])
    output_data = output_tensor.copy_to_cpu()

    print("Inference result of assets/infer_3.png is: ", np.argmax(output_data))


if __name__ == '__main__':
    infer_mnist(saved_model='assets/mnist', image_path='assets/infer_3.png')
