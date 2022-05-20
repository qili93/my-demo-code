#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.io
import paddle.static
import paddle.vision


# function to calculate top_k accurary
def top_k_accuracy(truths, preds, k):
    truths = truths.ravel()
    classes = np.unique(truths, return_inverse=False)
    y_true_encoded = np.searchsorted(classes, truths)
    sorted_pred = np.argsort(preds, axis=1)[:, ::-1]
    hits = (y_true_encoded == sorted_pred[:, :k].T).any(axis=0)
    return np.average(hits)


# logger func for IPU Graph Compile
g_current_progress = 0
def ProgressFunc(progress, total):
    global g_current_progress
    if progress != g_current_progress and progress% 20 == 0:
        g_current_progress = progress
        print(f"Graph compilation: {progress}/{total}")


class MNIST:
    def __init__(self, batch_size=1, batches_per_step=64, cls=10):
        self.batch_size = batch_size
        self.bps = batches_per_step
        self.cls = cls

    # build model
    def build_model(self):
        self.img = paddle.static.data(
            name='img', shape=[self.batch_size, 1, 28, 28], dtype='float32')
        self.label = paddle.static.data(
            name='label', shape=[self.batch_size, 1], dtype='int64')
        conv_pool_1 = fluid.nets.simple_img_conv_pool(
            self.img, 20, 5, 2, 2, act="relu")
        conv_pool_bn_1 = fluid.layers.batch_norm(conv_pool_1)
        conv_pool_2 = fluid.nets.simple_img_conv_pool(
            conv_pool_bn_1, self.batch_size, 5, 2, 2, act="relu")
        self.prediction = fluid.layers.fc(conv_pool_2, self.cls, act='softmax')
        loss = fluid.layers.cross_entropy(self.prediction, self.label)
        self.loss = fluid.layers.mean(loss)

    # train function
    def train(self, exec, program, data_loader, run_ipu):
        print('start training')
        feed_list = [self.img.name, self.label.name]
        fetch_list = [self.prediction.name, self.loss.name]

        # set ipu strategy
        if run_ipu:
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(is_training=True)
            ipu_strategy.set_pipelining_config(batches_per_step=self.bps)
            ipu_strategy.set_options({
                "compilation_progress_logger": ProgressFunc}
            )
            print(f"start compiling model for ipu, it will need some minutes")
            ipu_compiler = paddle.static.IpuCompiledProgram(
                program, ipu_strategy=ipu_strategy)
            main_program = ipu_compiler.compile(feed_list, fetch_list)
            print(f"finish model compiling!")
        else:
            main_program = program

        for step, (img, label) in enumerate(data_loader()):
            _, loss = exec.run(
                main_program,
                feed={self.img.name: img,
                    self.label.name: label},
                fetch_list=fetch_list)
            if step % 40 == 0:
                print(f"step: {step}, loss: {np.mean(loss)}")
        print('finish training!')
        # weights to host
        ipu_compiler._backend.weights_to_host()

    # eval function
    def test(self, exec, program, data_loader, run_ipu):
        print('start verifying')
        feed_list = [self.img.name, self.label.name]
        fetch_list = [self.prediction.name, self.loss.name]

        # set ipu strategy
        if run_ipu:
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(is_training=False)
            ipu_strategy.set_pipelining_config(batches_per_step=self.bps)
            print(f"start compiling model for ipu, it will need some minutes")
            main_program = paddle.static.IpuCompiledProgram(
                program, ipu_strategy=ipu_strategy).compile(feed_list,
                                                            fetch_list)
            print(f"finish model compiling!")
        else:
            main_program = program
        
        preds, labels = [], []
        for img, label in data_loader():
            pred = exec.run(program=main_program,
                            feed={self.img.name: img,
                                  self.label.name: label},
                            fetch_list=fetch_list)
            preds.append(pred[0])
            labels.append(np.asarray(label))
        top1 = top_k_accuracy(np.vstack(labels), np.vstack(preds).reshape(-1, self.cls), k=1)
        print(f"top1 score: {top1}")


# prepare training data
def prepare_data(batch_size):
    transform = lambda img: (np.array(img) / 127.5 - 1.0).astype('float32')

    def collate_fn(data):
        imgs = np.expand_dims(np.asarray([x[0] for x in data]), axis=1)
        labels = np.asarray([x[1] for x in data])
        return imgs, labels

    train_loader = paddle.io.DataLoader(
        paddle.vision.datasets.MNIST(
            mode='train', transform=transform),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True)
    test_loader = paddle.io.DataLoader(
        paddle.vision.datasets.MNIST(
            mode='test', transform=transform),
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=True)
    return train_loader, test_loader


if __name__ == "__main__":
    paddle.enable_static()

    # build model
    mnist = MNIST()
    mnist.build_model()

    # create static program
    startup_program = paddle.static.default_startup_program()
    main_program = paddle.static.default_main_program()
    validation_program = main_program.clone(for_test=True)

    run_ipu = True  # use IPU or CPU
    place = paddle.IPUPlace() if run_ipu else paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    optimizer = paddle.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(mnist.loss)
    exe.run(startup_program)

    # train and eval
    train_loader, test_loader = prepare_data(mnist.batch_size * mnist.bps)
    mnist.train(exe, main_program, train_loader, run_ipu)
    mnist.test(exe, validation_program, test_loader, run_ipu)

    # save inference model
    paddle.static.save_inference_model('./model/mnist', 
                                      [mnist.img], 
                                      [mnist.prediction], 
                                      exe)
