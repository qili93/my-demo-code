import torch
import torch.npu
import torch.nn as nn

layer = nn.BatchNorm3d(4).to("npu:0")
layer.weight = torch.nn.Parameter(torch.ones(4).to("npu:0"))
layer.bias = torch.nn.Parameter(torch.ones(4).to("npu:0"))

input = torch.ones(2, 4, 6, 6, 6).to("npu:0")

# input.data = input.data.npu_format_cast(3) # ACL_FORMAT_NC1HWC0

output = layer(input)

print(output)
