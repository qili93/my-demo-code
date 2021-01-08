from __future__ import print_function

import numpy as np


INFER_OUT_PATH="../train-py/dconv08-out.txt"
LITE_OUT_PATH="../shell/lite-out-0.txt"

def compare_output():
    infer_out = np.loadtxt(INFER_OUT_PATH, dtype=np.float32)
    # infer_out.resize(OUTPUT_LENGTH)
    print("infer_out.shape=", infer_out.shape)

    lite_out = np.loadtxt(LITE_OUT_PATH, dtype=np.float32)
    # lite_out.resize(OUTPUT_LENGTH)
    print("lite_out.shape=", lite_out.shape)

    data_lenght = lite_out.shape[0]

    print("data length = ", data_lenght)

    for index in range(data_lenght):
        infer_res = infer_out[index]
        lite_res = lite_out[index]
        if (np.abs(infer_res - lite_res) > 0.002):
            print("abs error exceeded: index {}, infer res is {}, lite res is {}".format(index, infer_res, lite_res))

if __name__ == '__main__':
    compare_output()