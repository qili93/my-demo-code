from __future__ import print_function

import numpy as np


INFER_OUT_PATH="infer-out.raw"
LITE_OUT_PATH="lite-out.raw"

OUTPUT_LENGTH = 1 * 2 * 192 * 256

def compare_output():
    infer_out = np.fromfile(INFER_OUT_PATH, dtype=np.float32)
    infer_out.resize(OUTPUT_LENGTH)
    print("infer_out.shape=", infer_out.shape)

    lite_out = np.fromfile(LITE_OUT_PATH, dtype=np.float32)
    lite_out.resize(OUTPUT_LENGTH)
    print("lite_out.shape=", lite_out.shape)

    for index in range(OUTPUT_LENGTH):
        infer_res = infer_out[index]
        lite_res = lite_out[index]
        if (np.abs(infer_res - lite_res) > 0.001):
            print("abs error exceeded: index {}, infer res is {}, lite res is {}".format(index, infer_res, lite_res))

if __name__ == '__main__':
    compare_output()