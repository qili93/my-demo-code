import torch
import torch_npu
import torch.nn.functional as F

input = torch.ones(1, 4, 5, 5).to("npu:0")
weight = torch.ones(8, 4, 3, 3).to("npu:0")
bias = torch.ones(1, 8, 1, 1).to("npu:0")
output = F.conv2d(input, weight, bias, padding=1)

print(output)
