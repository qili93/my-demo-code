import torch
import torch.onnx
import numpy as np

# format
float_formatter = "{:9.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# Define model
class TheModelClass(torch.nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=8,
                               out_channels=8,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=8,
                               bias=True)
        print("conv weight size = {}".format(self.conv1.weight.data.size()))
        self.conv1.weight.data = torch.full((8, 1, 3, 3), 1.0, dtype=torch.float32)
        print("conv  bias  size = {}".format(self.conv1.bias.data.size()))
        self.conv1.bias.data = torch.full((8,), 1.0, dtype=torch.float32).view(8)

    def forward(self, x):
        x = self.conv1(x)
        return x

# Initialize model
torch_model = TheModelClass()

# Print model's state_dict
print("======= Model's state_dict =======")
for param_tensor in torch_model.state_dict():
    print(param_tensor, "\t", torch_model.state_dict()[param_tensor].size())

print("======= Save Model to ONNX =======")
batch_size = 1    # just a random number
# set the model to inference mode
torch_model.eval()
# Input to the model
x = torch.randn(batch_size, 8, 64, 64, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "torch-conv-64.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
