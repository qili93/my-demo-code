import torch
import torch_npu
import torch.nn as nn

layer = nn.BatchNorm3d(6).to("npu:0")
layer.weight = torch.nn.Parameter(torch.ones(4).to("npu:0"))
layer.bias = torch.nn.Parameter(torch.ones(4).to("npu:0"))

input = torch.ones(2, 4, 6, 6, 6).to("npu:0") # ACL_FORMAT_NCDHW
input.data = input.data.npu_format_cast(32) # ACL_FORMAT_NDC1HWC0
output = layer(input)
print(output)
