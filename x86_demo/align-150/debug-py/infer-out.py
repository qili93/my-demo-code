from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np

MODEL_PATH="../assets/models/align150-fp32-dst"

paddle.enable_static()

def infer_model(model_path=MODEL_PATH):
    if model_path is None:
        return

    img_np = np.ones([1, 80, 4, 4]).astype('float32')
    # img_np = read_image()
    # img_np = read_rawfile()
    
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.executor.global_scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
                                                 model_path, exe, '__model__', '__params__')
        output, = exe.run(inference_program,
                         feed={feed_target_names[0]: img_np},
                         fetch_list=fetch_targets,
                         return_numpy=False)

    output = np.array(output)
    print("output.shape=", output.shape)
    with open("infer-out.raw", "wb") as f:
        output.tofile(f)

if __name__ == '__main__':
    infer_model(model_path=MODEL_PATH)
