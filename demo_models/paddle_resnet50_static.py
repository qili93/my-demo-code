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
import argparse
import time

EPOCH_NUM = 3
BATCH_NUM = 5005
BATCH_SIZE = 256

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'gpu', 'npu', 'ascend'],
        default="ascend",
        help="Choose the device to run, it can be: cpu/gpu/npu/ascend, default is ascend.")
    parser.add_argument(
        "--amp",
        action='store_true',
        default=True,
        help="Enable auto mixed precision training.")
    return parser.parse_args()

# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([3, 224, 224]).astype('float32')
        label = np.random.randint(0, 9, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

def main():
    args = parse_args()
    print('---------------  Running Arguments ---------------')
    print(args)
    print('--------------------------------------------------')

    # enable static and set device
    paddle.enable_static()
    paddle.set_device(args.device)

    # program
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    with paddle.static.program_guard(main_program, startup_program):
        # model
        # model = LeNet5()
        model = paddle.vision.models.resnet50()
        cost = nn.CrossEntropyLoss()
        # inputs
        images = static.data(shape=[None, 3, 224, 224], name='image', dtype='float32')
        labels = static.data(shape=[None, 1], name='label', dtype='int64')
        # foward
        outputs = model(images)
        loss = cost(outputs, labels)

        # optimizer and amp
        optimizer = paddle.optimizer.SGD(learning_rate=0.1,parameters=model.parameters())
        if args.amp:
            amp_list = paddle.static.amp.CustomOpLists(
                custom_black_list=["flatten_contiguous_range", "greater_than"])
            optimizer = paddle.static.amp.decorate(
                optimizer=optimizer,
                amp_lists=amp_list,
                init_loss_scaling=1024,
                use_dynamic_loss_scaling=True)
        optimizer.minimize(loss)

    # create data loader
    dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
    train_loader = paddle.io.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

    # Data loading code
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # train_loader = paddle.io.DataLoader(
    #     paddle.vision.datasets.ImageFolder(
    #         root='/datasets/ILSVRC2012/train', 
    #         transform=transforms.Compose([
    #             transforms.RandomResizedCrop(224),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             normalize,
    #     ])),
    #     batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # static executor
    exe = static.Executor()
    exe.run(startup_program)

    # train and eval
    total_step = len(train_loader)
    for epoch_id in range(EPOCH_NUM):
        # train
        epoch_start = time.time()
        for batch_id, (train_image, train_label) in enumerate(train_loader()):
            train_loss = exe.run(
                main_program,
                feed={images.name: train_image,
                      labels.name: train_label},
                fetch_list=[loss])
        epoch_end = time.time()
        print(f"Epoch ID: {epoch_id+1}, Train time: {(epoch_end - epoch_start) * 1000} ms, Loss: {float(train_loss[0])}")


if __name__ == '__main__':
    main()
