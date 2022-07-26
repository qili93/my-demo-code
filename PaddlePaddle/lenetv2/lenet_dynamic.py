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

import numpy as np

import paddle
import paddle.nn as nn
from paddle.vision.transforms import Compose, Normalize

EPOCH_NUM = 1
BATCH_SIZE = 64

class LeNet(nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


# set device
paddle.set_device("cpu")

# model
net = LeNet()
loss_fn = nn.CrossEntropyLoss()
opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# data loader
transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
train_loader = paddle.io.DataLoader(train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)
test_loader = paddle.io.DataLoader(test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# train
for epoch_id in range(EPOCH_NUM):
    # train for each epoch
    for batch_id, (image, label) in enumerate(train_loader()):
        out = net(image)
        acc = paddle.metric.accuracy(out, label, k=1)
        loss = loss_fn(out, label)
        loss.backward()

        if batch_id % 100 == 0:
            print("Epoch {} batch {:0>3d}: loss = {}, acc = {}".format(epoch_id, batch_id, loss.numpy(), acc.numpy()))

        opt.step()
        opt.clear_grad()
    # test for each epoch
    acc_list = []
    loss_list = []
    net.eval()
    for batch_id, (image, label) in enumerate(test_loader()):
        out = net(image)
        acc = paddle.metric.accuracy(out, label, k=1)
        loss = loss_fn(out, label)
        loss_list.append(loss.numpy())
        acc_list.append(acc.numpy())
    loss_value = np.array(loss_list).mean()
    acc_value = np.array(acc_list).mean()
    print("Test at epoch {}, test loss = {:.5f}, test acc = {:>7.2%}".format(epoch_id, loss_value, acc_value))
    net.train()

# save inference model
paddle.jit.save(net, "assets/lenet_dynamic",
    input_spec=[paddle.static.InputSpec(shape=[None, 1, 28, 28], dtype='float32')])
