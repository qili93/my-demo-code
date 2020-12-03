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
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               groups=8,
                               bias=True)
        print("conv weight size = {}".format(self.conv1.weight.data.size()))
        self.conv1.weight.data = torch.arange(1, 9, dtype=torch.float32).view(8, 1, 1, 1)
        # self.conv1.weight.data = torch.FloatTensor([10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001]).view(8, 1, 1, 1)
        print("conv weight data = \n{}".format(self.conv1.weight.data.detach().numpy()))
        print("conv bias size = {}".format(self.conv1.bias.data.size()))
        # self.conv1.bias.data = torch.full((8,), 0.0, dtype=torch.float32)
        self.conv1.bias.data = torch.arange(1, 9, dtype=torch.float32).view(8)
        # self.conv1.bias.data = torch.FloatTensor([1, 2, 3, 0.4, 5000, 600, 700, 800]).view(8)
        print("conv bias data = \n{}".format(self.conv1.bias.data.detach().numpy()))

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

input_data = np.empty([batch_size, 8, 2, 2], dtype=np.float32)
for ic in range(8):
    for ih in range(2):
        for iw in range(2):
            input_data[0, ic, ih, iw] = 100 * ic + 10 * ih + 1 * iw

x = torch.from_numpy(input_data) 
# x = torch.arange(1, 33, dtype=torch.float32).view(batch_size, 8, 2, 2)
print("conv input size = {}".format(x.size()))
print("conv input data = \n {}".format(x.detach().numpy()))
torch_out = torch_model(x)
print("conv output size = {}".format(torch_out.size()))
print("conv output data = \n {}".format(torch_out.detach().numpy()))

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "torch-conv-08.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
