#!/bin/bash

# git clone https://github.com/PaddlePaddle/PaddleClas.git -b develop

# prepare dataset of ILSVRC2012
cd PaddleClas
ln -s /datasets/ILSVRC2012 ./dataset/ILSVRC2012

# train with amp-o1 ultra
FLOG="paddleclas-resnet50-imagenet-card1-amp-o1-ultra-$(date +'%Y-%m-%d-%H-%M-%S').log"
echo "Start Time is: $(date +'%m/%d/%Y %T')" > ${FLOG}
python tools/train.py -c ppcls/configs/ImageNet/ResNet/ResNet50_amp_O1_ultra.yaml \
    -o Global.epochs=10 -o Global.eval_during_train=False \
    -o DataLoader.Train.sampler.batch_size=256 \
    -o DataLoader.Train.loader.num_workers=8 \
    -o AMP.level=O1 -o Global.save_interval=20 \
    -o Global.device=gpu >> ${FLOG} 2>&1 # edit device type to npu or xpu
echo "End Time is: $(date +'%m/%d/%Y %T')" >> ${FLOG}

