# https://github.com/pytorch/examples/blob/main/imagenet/main.py

import time
import argparse
import torch
import torch.npu
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from apex import amp

EPOCH_NUM = 3
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

def main():
    args = parse_args()
    print('---------------  Running Arguments ---------------')
    print(args)
    print('--------------------------------------------------')

    # set device
    torch.npu.set_device('npu:0')
    device = torch.device('npu:0')

    # set device to cuda
    # device = torch.device("cuda:0")

    # model
    # model = LeNet5().to(device)
    model = torchvision.models.resnet50().to(device)
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # conver to amp model
    model, optimizer = amp.initialize(model, optimizer, opt_level=AMP_LEVEL, loss_scale=1024, verbosity=1)

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('/datasets/ILSVRC2012/train', 
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
        ])),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=32, pin_memory=True, drop_last=True)

    # train
    iter_max = len(train_loader)
    for epoch_id in range(EPOCH_NUM):
        epoch_start = time.time()
        tic = time.time()
        for iter_id, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # get reader_cost
            reader_cost = time.time() - tic

            if args.graph:
                torch.npu.enable_graph_mode()
            
            #Forward pass
            outputs = model(images)
            # print(f"images={images.type(), images.size()}") # Float, [256, 3, 224, 224]
            # print(f"outputs={outputs.type(), outputs.size()}") # Half, [256, 1000]
            # print(f"labels={labels.type(), labels.size()}") # Long, [256]
            loss = cost(outputs, labels)
                
            # Backward and optimize
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            if args.graph:
                torch.npu.launch_graph()

            # batch_cost
            batch_cost = time.time() - tic
            # ips
            ips = BATCH_SIZE / batch_cost

            if (iter_id+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, reader_cost: {:.5f} s, batch_cost: {:.5f} s, ips: {:.5f} samples/s'
                       .format(epoch_id+1, EPOCH_NUM, iter_id+1, iter_max, loss.item(), reader_cost, batch_cost, ips))

            # update tic for each iter
            tic = time.time()

        if args.graph:
            torch.npu.disable_graph_mode()
            torch.npu.synchronize()
        
        epoch_cost = time.time() - epoch_start
        avg_ips = iter_max * BATCH_SIZE / epoch_cost
        print(f"Epoch ID: {epoch_id+1}, Train epoch time: {epoch_cost * 1000} ms, average ips: {avg_ips}")


if __name__ == '__main__':
    main()
