#!/bin/bash
set -ex


python3 pytorch_resnet50_imagenet.py --amp=O1 > pytorch_resnet50_imagenet_eager_amp_o1.log 2>&1

python3 pytorch_resnet50_imagenet.py --amp=O2 > pytorch_resnet50_imagenet_eager_amp_o2.log 2>&1

python3 pytorch_resnet50_imagenet.py --amp=O1 --graph > pytorch_resnet50_imagenet_graph_amp_o1.log 2>&1

python3 pytorch_resnet50_imagenet.py --amp=O2 --graph > pytorch_resnet50_imagenet_graph_amp_o2.log 2>&1

