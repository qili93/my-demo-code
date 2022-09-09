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
import paddle.static as static
import paddle.vision.transforms as transforms

import time

EPOCH_NUM = 5
BATCH_SIZE = 4096

class LeNet5(nn.Layer):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
          nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
          nn.BatchNorm2D(num_features=6),
          nn.ReLU(),
          nn.MaxPool2D(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
          nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
          nn.BatchNorm2D(num_features=16),
          nn.ReLU(),
          nn.MaxPool2D(kernel_size=2, stride=2))
        self.fc = nn.Linear(in_features=400, out_features=120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = paddle.flatten(out, start_axis=1, stop_axis=-1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# set device
paddle.enable_static()
paddle.set_device("npu:0")

# program
main_program = paddle.static.default_main_program()
startup_program = paddle.static.default_startup_program()

# model and loss
model = LeNet5()
cost = nn.CrossEntropyLoss()

# inputs
images = static.data(shape=[None, 1, 32, 32], name='image', dtype='float32')
labels = static.data(shape=[None, 1], name='label', dtype='int64')

# foward
outputs = model(images)
loss = cost(outputs, labels)

# optimizer and amp
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
amp_list = paddle.static.amp.CustomOpLists(custom_black_list=["flatten_contiguous_range", "greater_than"])
optimizer = paddle.static.amp.decorate(
    optimizer=optimizer,
    amp_lists=amp_list,
    init_loss_scaling=1024,
    use_dynamic_loss_scaling=True)
optimizer.minimize(loss)

# static executor
exe = static.Executor()
exe.run(startup_program)

# data loader
transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.1307,), std = (0.3081,))])
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform, download=True)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform, download=True)
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
total_step = len(train_loader)
for epoch_id in range(EPOCH_NUM):
    epoch_start = time.time()
    for batch_id, (train_image, train_label) in enumerate(train_loader()):
        train_loss = exe.run(main_program,
            feed={images.name: train_image,
                  labels.name: train_label},
            fetch_list=[loss.name],
            use_program_cache=True)
    epoch_end = time.time()
    print(f"Epoch ID: {epoch_id+1}, Train epoch time: {(epoch_end - epoch_start) * 1000} ms, Loss: {train_loss}")
