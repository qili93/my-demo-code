import torch
import torch.npu
import torch.nn as nn


device = torch.device('npu:0')
torch.npu.set_device("npu:0")

torch.npu.prof_init("./relu_model")
torch.npu.prof_start()

#Defining the convolutional neural network
class ReluModel(nn.Module):
    def __init__(self):
        super(ReluModel, self).__init__()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.relu(x)
        return out

model = ReluModel().to(device)

input = torch.randn(2).to(device)
out = model(input)
print(out)

torch.npu.prof_stop()
torch.npu.prof_finalize()
