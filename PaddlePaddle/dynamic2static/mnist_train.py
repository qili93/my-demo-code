#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import nn
import paddle.nn.functional as F

class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1):
        super().__init__()

        self.conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=None,
            bias_attr=False)
        self.bn = nn.BatchNorm(num_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MNIST(nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        self.conv0 = ConvBNLayer(
                    num_channels=1,
                    num_filters=4,
                    filter_size=5,
                    stride=1,
                    padding=0,
                    num_groups=1)
        self.conv1 = ConvBNLayer(
                    num_channels=4,
                    num_filters=4,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    num_groups=1)
        self.max_pool = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)
        self.fc = nn.Linear(in_features=144, out_features=10)

    def forward(self, inputs, label=None):
        x = self.inference(inputs)
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            loss = F.cross_entropy(x, label)
            avg_loss = paddle.mean(loss)
            return x, acc, avg_loss
        else:
            return x

    def inference(self, inputs):
        # [-1, 1, 28, 28]
        x = self.conv0(inputs)
        # [-1, 4, 24, 24]
        x1 = self.max_pool(x)
        # [-1, 4, 6, 6]
        x2 = self.conv1(x1)
        # [-1, 4, 6, 6]
        x = paddle.add(x=x1, y=x2)
        # [-1, 4, 6, 6]
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        # [-1, 144]
        x = self.fc(x)
        # [-1, 10]
        x = F.softmax(x)
        return x

def test_mnist(test_reader, mnist_model):
    acc_set = []
    avg_loss_set = []

    for batch_id, data in enumerate(test_reader()):
        x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)

        image = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)

        out, acc, avg_loss = mnist_model(image, label=label)

        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()
    return avg_loss_val_mean, acc_val_mean


def train_mnist(num_epochs, save_dirname):
    paddle.set_device('cpu')

    mnist = MNIST()
    mnist = paddle.jit.to_static(mnist)
    adam = paddle.optimizer.Adam(learning_rate=0.001, parameters=mnist.parameters())

    train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
    test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)

    for epoch in range(num_epochs):
        for batch_id, data in enumerate(train_reader()):
            x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
            y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)

            image = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)

            label.stop_gradient = True
            out, acc, avg_loss = mnist(image, label=label)
            avg_loss.backward()

            adam.minimize(avg_loss)
            mnist.clear_gradients()

            if batch_id % 100 == 0:
                print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))

        mnist.eval()
        test_cost, test_acc = test_mnist(test_reader, mnist)
        mnist.train()
        print("Epoch {} , Test avg_loss is: {}, acc is: {}".format(epoch, test_cost, test_acc))

    # save inference model
    if save_dirname is None:
        return
    # delete old model
    if  os.path.exists(save_dirname):
        shutil.rmtree(save_dirname)
        os.makedirs(save_dirname)
    # save inference model
    mnist.eval()
    paddle.jit.save(mnist, save_dirname)

if __name__ == '__main__':
    BATCH_SIZE = 64
    train_mnist(num_epochs=1, save_dirname='assets/mnist')
