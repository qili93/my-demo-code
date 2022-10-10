#!/bin/bash
set -ex

echo "======== Eager Mode ========"

python3 pytorch_resnet50_imagenet.py --amp=O0 > pytorch_resnet50_imagenet_eager_amp_o0.log 2>&1

sleep 10s

python3 pytorch_resnet50_imagenet.py --amp=O1 > pytorch_resnet50_imagenet_eager_amp_o1.log 2>&1

sleep 10s

python3 pytorch_resnet50_imagenet.py --amp=O2 > pytorch_resnet50_imagenet_eager_amp_o2.log 2>&1

sleep 10s

echo "======== Graph Mode ========"

python3 pytorch_resnet50_imagenet.py --amp=O0 --graph > pytorch_resnet50_imagenet_graph_amp_o0.log 2>&1

sleep 10s

python3 pytorch_resnet50_imagenet.py --amp=O1 --graph > pytorch_resnet50_imagenet_graph_amp_o1.log 2>&1

sleep 10s

python3 pytorch_resnet50_imagenet.py --amp=O2 --graph > pytorch_resnet50_imagenet_graph_amp_o2.log 2>&1

sleep 10s
