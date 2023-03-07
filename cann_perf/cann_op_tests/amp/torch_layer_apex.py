# reference: https://johnwlambert.github.io/conv-backprop/

import torch
import torch_npu
import torch.npu
import numpy as np
from apex import amp

model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)).to("npu:0")
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# conver to amp
model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale="dynamic", verbosity=1)
model.train()


x_tensor = torch.ones(4, 1, 24, 24).to("npu:0")
label = torch.ones(4, 6, 10, 10).to("npu:0")

out = model(x_tensor)
loss = cost(out, label)

with amp.scale_loss(loss, optimizer) as scaled_loss:
  scaled_loss.backward()

optimizer.step()
optimizer.zero_grad()
