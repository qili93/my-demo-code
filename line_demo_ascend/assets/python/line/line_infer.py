from __future__ import print_function

import os
import numpy
import paddle
import paddle.fluid as fluid

def save_result(points1, points2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x1 = [idx for idx in range(len(points1))]
    y1 = points1
    y2 = points2
    l1 = plt.plot(x1, y1, 'r--', label='predictions')
    l2 = plt.plot(x1, y2, 'g--', label='GT')
    plt.plot(x1, y1, 'ro-', x1, y2, 'g+-')
    plt.title('predictions VS GT')
    plt.legend()
    plt.savefig('./image/prediction_gt.png')


def infer_line(save_dirname):
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(3) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)
        # prepare inference data
        infer_reader = paddle.batch(paddle.dataset.uci_housing.test(), batch_size=10)
        infer_data = next(infer_reader())
        infer_feat = numpy.array([data[0] for data in infer_data]).astype("float32")
        infer_label = numpy.array([data[1] for data in infer_data]).astype("float32")

        assert feed_target_names[0] == 'x'
        results = exe.run(inference_program, feed={feed_target_names[0]: numpy.array(infer_feat)}, fetch_list=fetch_targets)

        print("infer results: (House Price)")
        for idx, val in enumerate(results[0]):
            print("%d: %.2f" % (idx, val))

        print("\nground truth:")
        for idx, val in enumerate(infer_label):
            print("%d: %.2f" % (idx, val))

        save_result(results[0], infer_label)

if __name__ == '__main__':
    infer_line(save_dirname='../../models/line_model')