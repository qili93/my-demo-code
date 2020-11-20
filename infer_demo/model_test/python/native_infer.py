from __future__ import print_function

import os
import numpy as np
import cv2
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid import core

MODEL_PATH='../assets/models/mobilenet_v1_fp32_224_fluid'

paddle.enable_static()

def infer_model():
    img_np = np.ones([1, 3, 224, 224]).astype('float32')

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.executor.global_scope()
    with fluid.scope_guard(inference_scope):
       # uncombined model
      [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
                                                 dirname=MODEL_PATH, executor=exe)

      output, = exe.run(inference_program,
                    feed={feed_target_names[0]: img_np},
                    fetch_list=fetch_targets,
                    return_numpy=False)
                    
      output = np.array(output)
      print("output.shape=", output.shape)

      # write output to file
    #   with open("infer-out.raw", "wb") as f:
    #     output.tofile(f)

if __name__ == '__main__':
    infer_model()