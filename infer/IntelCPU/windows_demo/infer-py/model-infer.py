from __future__ import print_function

import os
import numpy as np
import cv2
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid import core

MODEL_PATH='../assets/models/pc-seg-float-model'
# MODEL_PATH='../assets/models/pc-seg-float-model-dst'

# paddle.enable_static()

def infer_model():
    img_np = np.ones([1, 4, 192, 256]).astype('float32')

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.executor.global_scope()
    with fluid.scope_guard(inference_scope):
      [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
                                                 MODEL_PATH, exe, '__model__', '__params__')

      output, = exe.run(inference_program,
                    feed={feed_target_names[0]: img_np},
                    fetch_list=fetch_targets,
                    return_numpy=False)
                    
      output = np.array(output)
      print("output.shape=", output.shape)
      print("output.dtype=", output.dtype)

      # write output to file
      output.tofile("infer-out.raw")
        
      # write output to txt
      np.savetxt("infer-out.txt", output.flatten())

def compare_data():
  infer_raw = np.fromfile("infer-out.raw", dtype=np.float32)
  # infer_raw.resize(1, 2, 192, 256)
  infer_raw.resize(1 * 2 * 192 * 256)
  print("infer_raw shape:", infer_raw.shape)
  print("infer_raw dtype:", infer_raw.dtype)

  infer_txt = np.loadtxt('infer-out.txt', dtype=np.float32)
  # infer_txt.resize(1, 2, 192, 256)
  infer_raw.resize(1 * 2 * 192 * 256)
  print("infer_txt shape:", infer_txt.shape)
  print("infer_txt dtype:", infer_txt.dtype)

  assert np.allclose(infer_raw, infer_txt, 0.01)

  for i in range(400, 500):
    print("index {} == infer_raw data: {}, infer_txt data: {}".format(i, infer_raw[i], infer_txt[i]))

def test_write_data():
  data = np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5)).astype('float32')

  # write output to file
  data.tofile("data-out.raw")

  # write output to txt
  np.savetxt("data-out.txt", data.flatten())

def test_load_data():
  data_raw = np.fromfile("data-out.raw", dtype=np.float32)
  data_raw.resize(2, 3, 4, 5)
  print("data_raw shape:", data_raw.shape)
  print("data_raw dtype:", data_raw.dtype)
  print("data_raw data:", data_raw)

  data_txt = np.loadtxt('data-out.txt', dtype=np.float32)
  data_txt.resize(2, 3, 4, 5)
  print("data_txt shape:", data_txt.shape)
  print("data_txt dtype:", data_txt.dtype)
  print("data_txt data:", data_txt)

  assert np.allclose(data_raw, data_txt, 0.01)


if __name__ == '__main__':
    infer_model()
    compare_data()
    # test_write_data()
    # test_load_data()
    
