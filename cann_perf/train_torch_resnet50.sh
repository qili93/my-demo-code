#!/bin/bash
set -ex

DEVICE_TARGET=${1:-Ascend} # GPU, Ascend
DEVICE_ID=${2:-0} # device id

if [ "$DEVICE_TARGET" = "GPU" ];then
    sed -i "s/import torch.npu/#import torch.npu/g" $FILENAME
fi

echo "======== Eager Mode ========"
python3 torch_resnet50_imagenet.py --device=${DEVICE_TARGET} --ids=${DEVICE_ID} --amp=O0 > torch_resnet50_imagenet_eager_amp_o0.log 2>&1
sleep 10s
python3 torch_resnet50_imagenet.py --device=${DEVICE_TARGET} --ids=${DEVICE_ID} --amp=O1 > torch_resnet50_imagenet_eager_amp_o1.log 2>&1
sleep 10s
python3 torch_resnet50_imagenet.py --device=${DEVICE_TARGET} --ids=${DEVICE_ID} --amp=O2 > torch_resnet50_imagenet_eager_amp_o2.log 2>&1
sleep 10s


if [ "$DEVICE_TARGET" = "Ascend" ];then
    echo "======== Graph Mode ========"
    python3 torch_resnet50_imagenet.py --device=${DEVICE_TARGET} --ids=${DEVICE_ID} --amp=O0 --graph > torch_resnet50_imagenet_graph_amp_o0.log 2>&1
    sleep 10s
    python3 torch_resnet50_imagenet.py --device=${DEVICE_TARGET} --ids=${DEVICE_ID} --amp=O1 --graph > torch_resnet50_imagenet_graph_amp_o1.log 2>&1
    sleep 10s
    python3 torch_resnet50_imagenet.py --device=${DEVICE_TARGET} --ids=${DEVICE_ID} --amp=O2 --graph > torch_resnet50_imagenet_graph_amp_o2.log 2>&1
fi
