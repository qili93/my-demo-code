# https://github.com/pytorch/examples/blob/main/imagenet/main.py

import time
import argparse
import numpy as np
import torch
import torch.npu
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from apex import amp

EPOCH_NUM = 3
BATCH_NUM = 5005
BATCH_SIZE = 256
AMP_LEVEL = "O1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amp",
        action='store_true',
        default=True,
        help="Enable auto mixed precision training.")
    parser.add_argument(
        "--graph",
        action='store_true',
        default=False,
        help="Whether to perform graph mode in train")
    return parser.parse_args()


# define a random dataset
class RandomDataset(torch.utils.data.Dataset):
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

    # set device
    torch.npu.set_device('npu:0')
    device = torch.device('npu:0')

    # model
    # model = LeNet5().to(device)
    model = torchvision.models.resnet50().to(device)
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # conver to amp model
    model, optimizer = amp.initialize(model, optimizer, opt_level=AMP_LEVEL, loss_scale=1024, verbosity=1, combine_grad=False)

    # create data loader
    dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, )

    # train
    total_step = len(train_loader)
    for epoch_id in range(EPOCH_NUM):
        epoch_start = time.time()
        for batch_id, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            if args.graph:
                torch.npu.enable_graph_mode()
            
            #Forward pass
            outputs = model(images)
            loss = cost(outputs, labels)
                
            # Backward and optimize
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            if args.graph:
                torch.npu.launch_graph()

            if (batch_id+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch_id+1, EPOCH_NUM, batch_id+1, total_step, loss.item()))

        if args.graph:
            torch.npu.disable_graph_mode()
            torch.npu.synchronize()
        
        epoch_end = time.time()
        print(f"Epoch ID: {epoch_id+1}, Train epoch time: {(epoch_end - epoch_start) * 1000} ms")


if __name__ == '__main__':
    main()
