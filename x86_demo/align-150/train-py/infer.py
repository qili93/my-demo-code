from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np
import cv2
import math

# format
float_formatter = "{:9.1f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

MODEL_PATH="../assets/models/dconv08"
# paddle.enable_static()

def infer_model(model_path):
    if model_path is None:
        return

    input_data = np.ones([1, 8, 64, 64]).astype('float32')

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

    model_name = model_path.rpartition("/")[2]
    output1 = np.array(out1)
    print(model_name+" output 1 shape is : "+str(output1.shape))
    # save to txt file
    np.savetxt("infer-out-0.txt", output1.flatten(), fmt='%10.3f')
    # save to raw file
    # with open("infer-out-0.raw", "wb") as f:
    #     output1.tofile(f)

if __name__ == '__main__':
    infer_model(model_path=MODEL_PATH)