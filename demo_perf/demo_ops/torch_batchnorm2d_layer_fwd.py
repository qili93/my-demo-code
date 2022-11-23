import torch
import torch.npu
import torch.nn as nn

# aclopCompileAndExecute, opType = Identity
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [6], StorageShape = [1, 1, 1, 1, 16], shapeRange = [], memtype = 0, isConst = 0

# aclopCompileAndExecute, opType = Identity
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [6], StorageShape = [1, 1, 1, 1, 16], shapeRange = [], memtype = 0, isConst = 0

# aclopCompileAndExecute, opType = Identity
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [6], StorageShape = [1, 1, 1, 1, 16], shapeRange = [], memtype = 0, isConst = 0

# aclopCompileAndExecute, opType = Identity
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [6], StorageShape = [1, 1, 1, 1, 16], shapeRange = [], memtype = 0, isConst = 0

layer = nn.BatchNorm2d(6).to("npu:0")
layer.weight = torch.nn.Parameter(torch.ones(6).to("npu:0"))
layer.bias = torch.nn.Parameter(torch.ones(6).to("npu:0"))

input = torch.ones(4,6,24,24).to("npu:0")
# aclopCompileAndExecute, opType = Identity
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 6, 24, 24], StorageShape = [4, 6, 24, 24], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [4, 6, 24, 24], StorageShape = [4, 1, 24, 24, 16], shapeRange = [], memtype = 0, isConst = 0
input.data = input.data.npu_format_cast(3) # ACL_FORMAT_NC1HWC0


# aclopCompileAndExecute, opType = Add
# InputDesc[0]: [TensorDesc] DataType = 3, Format = 2, StorageFormat = -1, Shape = [], StorageShape = [], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 3, Format = 2, StorageFormat = 2, Shape = [], StorageShape = [], shapeRange = [], memtype = 2, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 3, Format = 2, StorageFormat = 2, Shape = [], StorageShape = [], shapeRange = [], memtype = 0, isConst = 0

# aclopCompileAndExecute, opType = Identity
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [6], StorageShape = [1, 1, 1, 1, 16], shapeRange = [], memtype = 0, isConst = 0

# aclopCompileAndExecute, opType = Identity
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [6], StorageShape = [1, 1, 1, 1, 16], shapeRange = [], memtype = 0, isConst = 0

# aclopCompileAndExecute, opType = BNTrainingReduce
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [4, 6, 24, 24], StorageShape = [4, 1, 24, 24, 16], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [6], StorageShape = [1, 1, 1, 1, 16], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[1]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [6], StorageShape = [1, 1, 1, 1, 16], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {epsilon = 1e-05}

# aclopCompileAndExecute, opType = BNTrainingUpdate
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [4, 6, 24, 24], StorageShape = [4, 1, 24, 24, 16], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[1]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [6], StorageShape = [1, 1, 1, 1, 16], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[2]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [6], StorageShape = [1, 1, 1, 1, 16], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[3]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [6], StorageShape = [1, 1, 1, 1, 16], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[4]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [6], StorageShape = [1, 1, 1, 1, 16], shapeRange = [], memtype = 0, isConst = 0 
# InputDesc[5]: [TensorDesc] DataType = 0, Format =

output = layer(input)

# aclopCompileAndExecute, opType = Identity
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [4, 6, 24, 24], StorageShape = [4, 1, 24, 24, 16], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 6, 24, 24], StorageShape = [4, 6, 24, 24], shapeRange = [], memtype = 0, isConst = 0
print(output)
