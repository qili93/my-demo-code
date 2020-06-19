#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import os
import shutil
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import TracedLayer

class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        self.conv1 = fluid.dygraph.Conv2D(1, 32, 3, 1)
        self.fc1 = fluid.dygraph.Linear(5408, 10)

    def forward(self, inputs, label=None):
        # [-1, 1, 28, 28]
        x = self.conv1(inputs)
        # [-1, 32, 26, 26]
        x = fluid.layers.relu(x)
        x = fluid.layers.pool2d(input=x, pool_size=2, pool_stride=2, pool_type='max')
        # [-1, 32, 13, 13]
        x = fluid.layers.flatten(x, 1)
        # [-1, 5408]
        x = self.fc1(x)
        # [-1, 10]
        output = fluid.layers.softmax(x)
        if label is not None:
            acc = fluid.layers.accuracy(input=x, label=label)
            return output, acc
        else:
            return output

def test_mnist(test_reader, mnist_model):
    acc_set = []
    avg_loss_set = []

    for batch_id, data in enumerate(test_reader()):
        dy_x_data = np.array(
            [x[0].reshape(1, 28, 28) for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)

        img = fluid.dygraph.base.to_variable(dy_x_data)
        label = fluid.dygraph.base.to_variable(y_data)

        prediction, acc = mnist_model(img, label)

        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = fluid.layers.mean(loss)

        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    return avg_loss_val_mean, acc_val_mean


def train_mnist(num_epochs):
    place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()

    with fluid.dygraph.guard(place):
        mnist = MNIST()

        adam = fluid.optimizer.AdamOptimizer(
            learning_rate=0.001, parameter_list=mnist.parameters())

        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)

        for epoch in range(num_epochs):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)

                img = fluid.dygraph.base.to_variable(dy_x_data)
                label = fluid.dygraph.base.to_variable(y_data)
                cost, acc = mnist(img, label)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()
                adam.minimize(avg_loss)

                mnist.clear_gradients()

                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(
                        epoch, batch_id, avg_loss.numpy()))

            mnist.eval()
            test_cost, test_acc = test_mnist(test_reader, mnist)
            mnist.train()
            print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format(epoch, test_cost, test_acc))

        # save inference model
        save_dirname = '../assets/models/simple_mnist'
        # delete old model
        shutil.rmtree(save_dirname)
        # os.makedirs(save_dirname, exist_ok=True)
        # save inference model
        in_np = np.random.random([1, 1, 28, 28]).astype('float32')
        input_var = fluid.dygraph.to_variable(in_np)
        out_dygraph, static_layer = TracedLayer.trace(mnist, inputs=[input_var])
        
        static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])
        print("Saved inference model to {}".format(save_dirname))

if __name__ == '__main__':
    BATCH_SIZE = 64
    train_mnist(num_epochs=5)