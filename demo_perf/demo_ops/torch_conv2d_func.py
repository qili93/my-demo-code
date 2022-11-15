import torch
import torch.npu
import torch.nn.functional as F

filters = torch.ones(8, 4, 3, 3).to("npu:0")
inputs = torch.ones(1, 4, 5, 5).to("npu:0")
bias = torch.ones(1, 8, 1, 1).to("npu:0")
output = F.conv2d(inputs, filters, bias, padding=1)

print(output)
