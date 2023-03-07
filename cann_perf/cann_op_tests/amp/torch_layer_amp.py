# reference: https://johnwlambert.github.io/conv-backprop/

import torch
import torch_npu
import torch.npu
# from torch_npu.npu.amp import GradScaler, autocast
import numpy as np
# from apex import amp

model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)).npu()
cost = torch.nn.CrossEntropyLoss().npu()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler = torch_npu.npu.amp.GradScaler(init_scale=2.)

x_tensor = torch.ones(4, 1, 24, 24).to("npu:0")
label = torch.ones(4, 6, 10, 10).to("npu:0")

with torch_npu.npu.amp.autocast():
  out = model(x_tensor)
  loss = cost(out, label)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

optimizer.zero_grad()
