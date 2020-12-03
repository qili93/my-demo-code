import torch
import torch.onnx
import numpy as np

# format
float_formatter = "{:9.1f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

model_name = "torch-conv_3x3s1.onnx"

# Define Conv Attr
# input
batch_size = 1
input_channel = 4
input_height = 8
input_width = 8
# filter
output_channel = 4
groups = 4
kernel_h = 3
kernel_w = 3
# attr
conv_stride = 1
conv_padding = 1
conv_dilation = 1
# output
output_height = 8
output_width = 8

# Init Conv Param
input_data = np.empty([batch_size, input_channel, input_height, input_width], dtype=np.float32)
for ic in range(input_channel):
    for ih in range(input_height):
        for iw in range(input_width):
            input_data[0, ic, ih, iw] = 100 * ic + 10 * ih + iw

# print(input_data) - {1, 4, 8, 8}
print("Input Data = [{}, {}, {}, {}]".format(batch_size, input_channel, input_height, input_width))
for ic in range(input_channel):
    for ih in range(input_height):
        print("[ {} ]".format(" ".join([str(float_formatter(v)) for v in input_data[0, ic, ih, :]])))
    print("")

filter_data = np.empty([output_channel, 1, kernel_h, kernel_w], dtype=np.float32)
for oc in range(output_channel):
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            filter_data[oc, 0, kh, kw] = 100 * oc + 10 * kh + kw

# print(filter_data) - {4, 1, 1, 1}
print("Filter Data = [{}, {}, {}, {}]".format(output_channel, 1, kernel_h, kernel_w))
for oc in range(output_channel):
    for kh in range(kernel_h):
        print("[ {} ]".format(" ".join([str(float_formatter(v)) for v in filter_data[oc, 0, kh, :]])))
    print("")

bias_data = np.empty([output_channel,], dtype=np.float32)
for oc in range(output_channel):
    # bias_data[oc] = oc
    bias_data[oc] = 0
# print(bias_data) - {4}
print("Bias Data = {4}")
print("[ {} ]".format(" ".join(str(float_formatter(v)) for v in bias_data)))
print("")

# Define model
class TheModelClass(torch.nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=input_channel,
                               out_channels=output_channel,
                               kernel_size=kernel_h,
                               stride=conv_stride,
                               padding=conv_padding,
                               groups=groups,
                               bias=True)
        print("conv weight size = {}".format(self.conv1.weight.data.size()))
        self.conv1.weight.data = torch.from_numpy(filter_data)
        # self.conv1.weight.data = torch.arange(1, 9, dtype=torch.float32).view(4, 1, 3, 3)
        # self.conv1.weight.data = torch.FloatTensor([10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001]).view(8, 1, 1, 1)
        # print("conv weight data = \n{}".format(self.conv1.weight.data.detach().numpy()))
        print("conv bias size = {}".format(self.conv1.bias.data.size()))
        self.conv1.bias.data = torch.from_numpy(bias_data)
        # self.conv1.bias.data = torch.full((8,), 0.0, dtype=torch.float32)
        # self.conv1.bias.data = torch.arange(1, 9, dtype=torch.float32).view(8)
        # self.conv1.bias.data = torch.FloatTensor([1, 2, 3, 0.4, 5000, 600, 700, 800]).view(8)
        # print("conv bias data = \n{}".format(self.conv1.bias.data.detach().numpy()))

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
# set the model to inference mode
torch_model.eval()
# Input to the model
x = torch.from_numpy(input_data) 
# x = torch.arange(1, 33, dtype=torch.float32).view(batch_size, 8, 2, 2)
print("conv input size = {}".format(x.size()))
# print("conv input data = \n {}".format(x.detach().numpy()))
torch_out = torch_model(x)
print("conv output size = {}".format(torch_out.size()))
# print("conv output data = \n {}".format(torch_out.detach().numpy()))

# print(output_data) - {1, 4, 8, 8}
output_data = torch_out.detach().numpy()
print("Output Data = {1, 4, 8, 8}")
for oc in range(output_channel):
    for oh in range(output_height):
        print("[ {} ]".format(" ".join([str(float_formatter(v)) for v in output_data[0, oc, oh, :]])))
    print("")

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  model_name,                # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
