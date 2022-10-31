import torch
import torch.npu
import torch.nn as nn

# # layer = nn.Sequential(
# #             nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
# #             nn.BatchNorm2d(6),
# #             nn.ReLU(),
# #             nn.MaxPool2d(kernel_size = 2, stride = 2)).to("npu:0")


# # layer = nn.Sequential(
# #             nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0), # 4,1,28,28
# #             nn.BatchNorm2d(6)).to("npu:0") # 4,6,24,24
# # bn = nn.BatchNorm2d(6).to("npu:0")
# input = torch.full(size=(4, 1, 28, 28), fill_value=1.0, device="npu:0")
# output = layer(input)
# print(output.size())


# bn = nn.BatchNorm2d(6)
bn = nn.BatchNorm2d(6).to("npu:0")
input = torch.ones(4,6,24,24).to("npu:0")
input.data = input.data.npu_format_cast(3)
output = bn(input)
print(output.size())