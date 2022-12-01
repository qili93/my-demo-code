import paddle
import paddle.nn as nn

paddle.set_device("npu")


# aclopCompileAndExecute, opType = Fills
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {value = 1}

# aclopCompileAndExecute, opType = Fills
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {value = 1}

weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0))
bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0))

# aclopCompileAndExecute, opType = Fills
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {value = 0}

# aclopCompileAndExecute, opType = Fills
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {value = 1}

##### REPEAT 4 TIMES #######
# aclopCompileAndExecute, opType = Identity
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [6], StorageShape = [6], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [6], StorageShape = [1, 1, 1, 1, 16], shapeRange = [], memtype = 0, isConst = 0

layer = nn.BatchNorm2D(num_features=6, weight_attr=weight_attr, bias_attr=bias_attr)

# aclopCompileAndExecute, opType = Fills
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 6, 24, 24], StorageShape = [4, 6, 24, 24], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 6, 24, 24], StorageShape = [4, 6, 24, 24], shapeRange = [], memtype = 0, isConst = 0 
# Attr: {value = 1}
input = paddle.ones(shape=[4, 6, 24, 24])

# aclopCompileAndExecute, opType = Identity
# InputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 0, Shape = [4, 6, 24, 24], StorageShape = [4, 6, 24, 24], shapeRange = [], memtype = 0, isConst = 0 
# OutputDesc[0]: [TensorDesc] DataType = 0, Format = 0, StorageFormat = 3, Shape = [4, 6, 24, 24], StorageShape = [4, 1, 24, 24, 16], shapeRange = [], memtype = 0, isConst = 0
input = paddle.incubate._npu_identity(x=input, format=3) # ACL_FORMAT_NC1HWC0

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
