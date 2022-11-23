import torch
import torch.npu
import torch.nn.functional as F


input = torch.ones(4, 6, 24, 24).to("npu:0")
weight = torch.ones(6).to("npu:0")
bias = torch.ones(6).to("npu:0")
running_mean = torch.ones(6).to("npu:0")
running_var = torch.ones(6).to("npu:0")

# output = F.batch_norm(input, weight, bias, running_mean, running_var, training=False) # BNInfer

output = F.batch_norm(input, weight, bias, running_mean, running_var, training=True)

print(output)
