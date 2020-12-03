x2paddle --framework=onnx \
         --model=torch-conv_1x1s1.onnx \
         --save_dir=torch-conv_1x1s1
# input with -1,4,8,8

x2paddle --framework=onnx \
         --model=torch-conv_3x3s1.onnx \
         --save_dir=torch-conv_3x3s1
# input with -1,4,8,8