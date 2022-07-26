import os
import numpy as np

import paddle
import paddle.nn as nn
import paddle.static as static
from paddle.vision.transforms import Compose, Normalize
from paddle.utils.cpp_extension import load

os.environ['CPU_NUM'] = '4'
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
paddle.enable_static()
paddle.set_device("cpu")

# model
image  = static.data(shape=[None, 1, 28, 28], name='image', dtype='float32')
label = static.data(shape=[None, 1], name='label', dtype='int64')

net = LeNet()
out = net(image)
acc = paddle.metric.accuracy(out, label, k=1)
loss = nn.functional.cross_entropy(out, label)

opt = paddle.optimizer.Adam(learning_rate=0.001)
opt.minimize(loss)

# data loader
transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
train_loader = paddle.io.DataLoader(train_dataset,
    feed_list=[image, label],
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)
test_loader = paddle.io.DataLoader(test_dataset,
    feed_list=[image, label],
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# prepare
exe = static.Executor()
exe.run(static.default_startup_program())

build_strategy = static.BuildStrategy()
build_strategy.debug_graphviz_path = "./graph"

places = paddle.static.cpu_places()
compiled_prog = static.CompiledProgram(
    static.default_main_program()).with_data_parallel(
        loss_name=loss.name, build_strategy=build_strategy, places=places)

# train
for epoch_id in range(EPOCH_NUM):
    # train for each epoch
    for batch_id, (image_data, label_data) in enumerate(train_loader()):
        loss_data, acc_data = exe.run(compiled_prog,
            feed={'image': image_data,
                  'label': label_data},
            fetch_list=[loss, acc])
        if batch_id % 100 == 0:
            print("Epoch {} batch {:0>3d}: loss = {:.5f}, acc = {:>7.2%}".format(epoch_id, batch_id, np.mean(loss_data), np.mean(acc_data)))
    # test for each epoch
    acc_list = []
    loss_list = []
    for batch_id, (image_data, label_data) in enumerate(test_loader()):
        loss_data, acc_data = exe.run(compiled_prog,
            feed={'image': image_data,
                  'label': label_data},
            fetch_list=[loss, acc])
        loss_list.append(np.mean(loss_data))
        acc_list.append(np.mean(acc_data))
    loss_value = np.array(loss_list).mean()
    acc_value = np.array(acc_list).mean()
    print("Test at epoch {}, test loss = {:.5f}, test acc = {:>7.2%}".format(epoch_id, loss_value, acc_value))

# save inference model
static.save_inference_model("assets/lenet_static", [image], [out], exe)
