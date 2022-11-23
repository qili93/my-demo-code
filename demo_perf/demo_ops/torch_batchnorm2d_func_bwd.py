import torch
import torch.npu
import torch.nn.functional as F

# ERROR: RuntimeError: the derivative for 'running_mean' is not implemented


input = torch.ones(4, 6, 24, 24).to("npu:0")
weight = torch.ones(6).to("npu:0")
bias = torch.ones(6).to("npu:0")
input.requires_grad = True
weight.requires_grad = True
bias.requires_grad = True
running_mean = torch.ones(6).to("npu:0")
running_var = torch.ones(6).to("npu:0")
running_mean.requires_grad = False
running_var.requires_grad = False

# output = F.batch_norm(input, weight, bias, running_mean, running_var, training=False) # BNInfer

output = F.batch_norm(input, weight, bias, running_mean, running_var, training=True)

loss = out.sum()

loss.backward()

print("input.grad=", input.grad)
print("weight.grad=", weight.grad)
print("bias.grad=", bias.grad)
