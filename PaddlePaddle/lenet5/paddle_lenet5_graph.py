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

EPOCH_NUM = 1
BATCH_SIZE = 4096


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'gpu', 'npu'],
        default="npu",
        help="Choose the device to run, it can be: cpu/gpu/npu, default is npu.")
    parser.add_argument(
        '--ids',
        type=int,
        default=0,
        help="Choose the device id to run, default is 0.")
    parser.add_argument(
        "--amp",
        action='store_true',
        default=False,
        help="Enable auto mixed precision training.")
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation in train")
    parser.add_argument(
        "--infer",
        action='store_true',
        default=False,
        help="Whether to perform inference after train")
    return parser.parse_args()


class LeNet5(nn.Layer):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2D(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=0),
            nn.BatchNorm2D(num_features=6),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2D(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0),
            nn.BatchNorm2D(num_features=16),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=2, stride=2))
        self.fc = nn.Linear(in_features=400, out_features=120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = paddle.flatten(out, 1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


def infer_func(saved_model, device_type=None):
    # create config
    config = paddle.inference.Config(saved_model + '.pdmodel',
                                     saved_model + '.pdiparams')
    # enable custom device
    if device_type == "npu":
        config.enable_custom_device("npu")
    elif device_type == "gpu":
        config.enable_use_gpu(100, 0)
    else:
        config.disable_gpu()  # use cpu

    # create predictor
    predictor = paddle.inference.create_predictor(config)
    # random input
    input_data = np.random.random(size=[1, 1, 32, 32]).astype('float32')
    # set input
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])
    input_tensor.reshape(input_data.shape)
    input_tensor.copy_from_cpu(input_data.copy())
    # Run
    predictor.run()
    # Set output
    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])
    output_data = output_tensor.copy_to_cpu()
    print("Inference result is: ", np.argmax(output_data))


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
        model = LeNet5()
        cost = nn.CrossEntropyLoss()
        # inputs
        images = static.data(
            shape=[None, 1, 32, 32], name='image', dtype='float32')
        labels = static.data(shape=[None, 1], name='label', dtype='int64')
        # foward
        outputs = model(images)
        loss = cost(outputs, labels)
        # accuracy
        acc_top1 = paddle.metric.accuracy(input=outputs, label=labels, k=1)
        acc_top5 = paddle.metric.accuracy(input=outputs, label=labels, k=5)

        # optimizer and amp
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=model.parameters())
        if args.amp:
            amp_list = paddle.static.amp.CustomOpLists(
                custom_black_list=["flatten_contiguous_range", "greater_than"])
            optimizer = paddle.static.amp.decorate(
                optimizer=optimizer,
                amp_lists=amp_list,
                init_loss_scaling=1024,
                use_dynamic_loss_scaling=True)
        optimizer.minimize(loss)

        # copy test program
        test_program = main_program.clone(for_test=True)

    # data loader
    transform = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307, ), std=(0.3081, ))
    ])
    train_dataset = paddle.vision.datasets.MNIST(
        mode='train', transform=transform, download=True)
    test_dataset = paddle.vision.datasets.MNIST(
        mode='test', transform=transform, download=True)
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2)
    test_loader = paddle.io.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2)

    # static executor
    place = paddle.CustomPlace("npu", 0)
    exe = static.Executor(place)
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
            # for debug
            # break
        epoch_end = time.time()
        print(
            f"Epoch ID: {epoch_id+1}, Train time: {(epoch_end - epoch_start) * 1000} ms, Loss: {float(train_loss[0])}"
        )

        # evaluation
        if args.eval:
            avg_acc = [[], []]
            for batch_id, (test_image, test_label) in enumerate(test_loader()):
                test_acc1, test_acc5 = exe.run(
                    test_program,
                    feed={images.name: test_image,
                        labels.name: test_label},
                    fetch_list=[acc_top1, acc_top5])
                avg_acc[0].append(float(test_acc1))
                avg_acc[1].append(float(test_acc5))
            print(
                f"Epoch ID: {epoch_id+1}, Top1 acc: {np.array(avg_acc[0]).mean()}, Top5 acc: {np.array(avg_acc[1]).mean()}"
            )

    # save inference model
    paddle.static.save_inference_model('build/lenet5', [images], [outputs], exe)

    # inference
    if args.infer:
        infer_func('build/lenet5', device_type=args.device)


if __name__ == '__main__':
    main()
