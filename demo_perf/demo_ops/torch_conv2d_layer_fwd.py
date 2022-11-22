import torch
import torch.npu
import torch.nn as nn
import torch.nn.functional as F

conv_2d = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0).to("npu:0")
input = torch.ones(4, 1, 28, 28).to("npu:0")
output = conv_2d(input)
print(output)
