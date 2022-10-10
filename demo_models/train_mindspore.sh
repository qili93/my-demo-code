#!/bin/bash
set -ex

DEVICE_TARGET=${1:-Ascend} # CPU, GPU, Ascend

echo "======== Eager Mode ========"
python3 mindspore_resnet50_imagenet.py --device=${DEVICE_TARGET} --amp=O0> mindspore_resnet50_imagenet_eager_amp_o0.log 2>&1
sleep 10s
python3 mindspore_resnet50_imagenet.py --device=${DEVICE_TARGET} --amp=O2 > mindspore_resnet50_imagenet_eager_amp_o2.log 2>&1
sleep 10s
python3 mindspore_resnet50_imagenet.py --device=${DEVICE_TARGET} --amp=O3 > mindspore_resnet50_imagenet_eager_amp_o3.log 2>&1
sleep 10s


echo "======== Graph Mode ========"
python3 mindspore_resnet50_imagenet.py --device=${DEVICE_TARGET} --amp=O0 --graph > mindspore_resnet50_imagenet_graph_amp_o0.log 2>&1
sleep 10s
python3 mindspore_resnet50_imagenet.py --device=${DEVICE_TARGET} --amp=O2 --graph > mindspore_resnet50_imagenet_graph_amp_o2.log 2>&1
sleep 10s
python3 mindspore_resnet50_imagenet.py --device=${DEVICE_TARGET} --amp=O3 --graph > mindspore_resnet50_imagenet_graph_amp_o3.log 2>&1
