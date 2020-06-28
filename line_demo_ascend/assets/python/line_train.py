from __future__ import print_function

import sys, os
import shutil
import math
import numpy
import paddle
import paddle.fluid as fluid


# For training test cost
def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(
            program=program, feed=feeder.feed(data_test), fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]
        count += 1
    return [x_d / count for x_d in accumulated]

def train_line(num_epochs, save_dirname):
    place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()

    train_reader = paddle.batch(paddle.dataset.uci_housing.train(), batch_size=BATCH_SIZE)
    test_reader = paddle.batch(paddle.dataset.uci_housing.test(), batch_size=BATCH_SIZE)

    # feature vector of length 13
    x = fluid.data(name='x', shape=[None, 13], dtype='float32')
    y = fluid.data(name='y', shape=[None, 1], dtype='float32')

    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(cost)

    test_program = main_program.clone(for_test=True)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_loss)

    exe = fluid.Executor(place)

    # main train loop.
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    exe.run(startup_program)
    exe_test = fluid.Executor(place)

    step_id = 0
    for pass_id in range(num_epochs):
        for data_train in train_reader():
            avg_loss_value, = exe.run(main_program, feed=feeder.feed(data_train),  fetch_list=[avg_loss])
            if step_id % 10 == 0:  # record a train cost every 100 batches
                print("%s, Step %d, Cost %f" %  ("Train cost", step_id, avg_loss_value[0]))

            if step_id % 100 == 0:  # record a test cost every 1000 batches
                test_metics = train_test(executor=exe_test, program=test_program, reader=test_reader, fetch_list=[avg_loss], feeder=feeder)
                print("%s, Step %d, Cost %f" %  ("===Test cost===", step_id, test_metics[0]))
                # If the accuracy is good enough, we can stop the training.
                if test_metics[0] < 10.0:
                    break

            step_id += 1

            if math.isnan(float(avg_loss_value[0])):
                sys.exit("got NaN loss, training failed.")

    # save inference model
    if save_dirname is None:
        return
    # delete old model
    if  os.path.exists(save_dirname):
        shutil.rmtree(save_dirname)
        os.makedirs(save_dirname)
    # We can save the trained parameters for the inferences later
    fluid.io.save_inference_model(save_dirname, ['x'], [y_predict], exe)
    print("Saved inference model to {}".format(save_dirname))

if __name__ == '__main__':
    BATCH_SIZE = 64
    train_line(num_epochs=100, save_dirname='../models/line_model')