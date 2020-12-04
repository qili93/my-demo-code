

# MODEL_NAME=dconv03 # -1,3,2,2
# MODEL_NAME=dconv04 # -1,4,2,2
# MODEL_NAME=dconv08 # -1,8,2,2
MODEL_NAME=dconv16 # -1,16,2,2

x2paddle --framework=onnx \
         --model=${MODEL_NAME}.onnx \
         --save_dir=${MODEL_NAME}