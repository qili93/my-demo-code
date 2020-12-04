x2paddle --framework=onnx \
         --model=conv3_1x1s1.onnx \
         --save_dir=conv3_1x1s1
# input with -1,3,2,2

x2paddle --framework=onnx \
         --model=conv4_1x1s1.onnx \
         --save_dir=conv4_1x1s1
# input with -1,4,2,2

x2paddle --framework=onnx \
         --model=conv8_1x1s1.onnx \
         --save_dir=conv8_1x1s1
# input with -1,8,2,2