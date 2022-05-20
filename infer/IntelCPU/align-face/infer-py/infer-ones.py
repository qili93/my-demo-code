from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np
import cv2
import math

MODEL_ALIGN = "../assets/models/align150-fp32"; # {1, 3, 128, 128}
INPUT_ALIGN = np.ones([1, 3, 128, 128]).astype('float32')

MODEL_EYES = "../assets/models/eyes_position-fp32"; # {1, 3, 32, 32}
INPUT_EYES = np.ones([1, 3, 32, 32]).astype('float32')

MODEL_IRIS = "../assets/models/iris_position-fp32"; # {1, 3, 24, 24}
INPUT_IRIS = np.ones([1, 3, 24, 24]).astype('float32')

MODEL_MOUTH = "../assets/models/mouth_position-fp32"; # {1, 3, 48, 48}
INPUT_MOUTH = np.ones([1, 3, 48, 48]).astype('float32')

paddle.enable_static()

def infer_align(model_path, input_data):
    if model_path is None:
        return
    
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.executor.global_scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
                                                 model_path, exe, '__model__', '__params__')
        out1, out2 = exe.run(inference_program,
                        feed={feed_target_names[0]: input_data},
                        fetch_list=fetch_targets,
                        return_numpy=False)

    model_name = model_path.rpartition("/")[2]
    output1 = np.array(out1)
    print(model_name+" output 1 shape is : "+str(output1.shape))
    np.savetxt(model_name+"-out1.txt", output1.flatten(), fmt='%10.3f')
    with open(model_name+"-out1.raw", "wb") as f:
        output1.tofile(f)

    output2 = np.array(out2)
    print(model_name+" output 2 shape is : "+str(output2.shape))
    np.savetxt(model_name+"-out2.txt", output1.flatten(), fmt='%10.3f')
    with open(model_name+"-out2.raw", "wb") as f:
        output1.tofile(f)


def infer_model(model_path, input_data):
    if model_path is None:
        return

    # img_np = np.ones([1, 3, 128, 128]).astype('float32')
    # img_np, M = read_image(IMAGE_FILE_PATH)
    # img_np, M = read_raw_file()
    
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.executor.global_scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
                                                 model_path, exe, '__model__', '__params__')
        out1 = exe.run(inference_program,
                        feed={feed_target_names[0]: input_data},
                        fetch_list=fetch_targets,
                        return_numpy=True)
    
    output1 = np.array(out1)
    model_name = model_path.rpartition("/")[2]
    print(model_name+" output shape is : "+str(output1.shape))

    np.savetxt(model_name+"-out1.txt", output1.flatten(), fmt='%10.3f')
    with open(model_name+"-out1.raw", "wb") as f:
        output1.tofile(f)

if __name__ == '__main__':
    infer_align(model_path=MODEL_ALIGN, input_data=INPUT_ALIGN)
    infer_model(model_path=MODEL_EYES, input_data=INPUT_EYES)
    infer_model(model_path=MODEL_IRIS, input_data=INPUT_IRIS)
    infer_model(model_path=MODEL_MOUTH, input_data=INPUT_MOUTH)
