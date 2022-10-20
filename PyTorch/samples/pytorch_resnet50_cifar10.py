# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import time
import argparse
import torch
import torch.npu
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from apex import amp

EPOCH_NUM = 5
BATCH_SIZE = 128
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # conver to amp model
    model, optimizer = amp.initialize(model, optimizer, opt_level=AMP_LEVEL, loss_scale=1024, verbosity=1, combine_grad=False)

    # data loader
    transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

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
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if args.graph:
                torch.npu.launch_graph()

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch_id+1, EPOCH_NUM, batch_id+1, total_step, loss.item()))

        if args.graph:
            torch.npu.disable_graph_mode()
            torch.npu.synchronize()
        
        epoch_end = time.time()
        print(f"Epoch ID: {epoch_id+1}, Train epoch time: {(epoch_end - epoch_start) * 1000} ms")


if __name__ == '__main__':
    main()