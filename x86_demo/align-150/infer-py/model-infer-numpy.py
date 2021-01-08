from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np

paddle.enable_static()

MODEL_PATH="../assets/models/align150-fp32"

# paddle.enable_static()

def infer_model(model_path):
    if model_path is None:
        return

    img_np = np.ones([1, 3, 128, 128]).astype('float32')
    # img_np, M = read_image(IMAGE_FILE_PATH)
    # img_np, M = read_raw_file()
    
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.executor.global_scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
                                                    model_path, exe, '__model__', '__params__')
        out1, out2 = exe.run(inference_program,
                        feed={feed_target_names[0]: img_np},
                        fetch_list=fetch_targets,
                        return_numpy=False)

    model_name = model_path.rpartition("/")[2]
    output1 = np.array(out1)
    print(model_name+" output 1 shape is : "+str(output1.shape))
    np.savetxt(model_name+"-out1.txt", output1.flatten(), fmt='%10.3f')
    # with open(model_name+"-out1.raw", "wb") as f:
    #     output1.tofile(f)

    output2 = np.array(out2)
    print(model_name+" output 2 shape is : "+str(output2.shape))
    np.savetxt(model_name+"-out2.txt", output2.flatten(), fmt='%10.3f')
    # with open(model_name+"-out2.raw", "wb") as f:
    #     output1.tofile(f)

if __name__ == '__main__':
    infer_model(model_path=MODEL_PATH)