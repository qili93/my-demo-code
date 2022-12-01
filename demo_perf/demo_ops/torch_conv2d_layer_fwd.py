import torch
import torch_npu
import torch.nn as nn

layer = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0).to("npu")
input = torch.ones(4, 1, 28, 28).to("npu")
output = layer(input)
print(output)
