#!/bin/bash
set -ex

# git clone https://github.com/PaddlePaddle/PaddleClas.git -b develop

# install dependency
pip install -r PaddleClas/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

# prepare dataset of ILSVRC2012
cd PaddleClas
ln -s /datasets/ILSVRC2012 ./dataset/ILSVRC2012

# GPU: train with amp-o1 ultra
export CUDA_VISIBLE_DEVICES=5
# export FLAGS_fraction_of_gpu_memory_to_use=0.80
# export FLAGS_cudnn_batchnorm_spatial_persistent=1
# export FLAGS_max_inplace_grad_add=8
# export FLAGS_cudnn_exhaustive_search=1
# export FLAGS_eager_delete_tensor_gb=0.0
# export FLAGS_conv_workspace_size_limit=4000
FLOG="paddleclas-resnet50-imagenet-card1-amp-o1-ultra-$(date +'%Y-%m-%d-%H-%M-%S').log"
echo "Start Time is: $(date +'%m/%d/%Y %T')" > ${FLOG}
python tools/train.py -c ppcls/configs/ImageNet/ResNet/ResNet50_amp_O1_ultra.yaml \
    -o Global.epochs=10 -o Global.eval_during_train=False \
    -o DataLoader.Train.sampler.batch_size=256 \
    -o DataLoader.Train.loader.num_workers=8 \
    -o AMP.level=O1 -o Global.save_interval=20 \
    -o Global.device=gpu \
    -o Global.use_dali=True \
    -o Global.to_static=True >> ${FLOG} 2>&1
echo "End Time is: $(date +'%m/%d/%Y %T')" >> ${FLOG}

# NPU: train with amp-o1 ultra
FLOG="paddleclas-resnet50-imagenet-card1-amp-o1-ultra-$(date +'%Y-%m-%d-%H-%M-%S').log"
echo "Start Time is: $(date +'%m/%d/%Y %T')" > ${FLOG}
python tools/train.py -c ppcls/configs/ImageNet/ResNet/ResNet50_amp_O1_ultra.yaml \
    -o Global.epochs=10 -o Global.eval_during_train=False \
    -o DataLoader.Train.sampler.batch_size=256 \
    -o DataLoader.Train.loader.num_workers=8 \
    -o AMP.level=O1 -o Global.save_interval=20 \
    -o Global.device=npu \
    -o Global.use_dali=False \
    -o Global.to_static=False >> ${FLOG} 2>&1 # edit device type to npu or xpu
echo "End Time is: $(date +'%m/%d/%Y %T')" >> ${FLOG}

# XPU: train with amp-o1 ultra
FLOG="paddleclas-resnet50-imagenet-card1-amp-o1-ultra-$(date +'%Y-%m-%d-%H-%M-%S').log"
echo "Start Time is: $(date +'%m/%d/%Y %T')" > ${FLOG}
python tools/train.py -c ppcls/configs/ImageNet/ResNet/ResNet50_amp_O1_ultra.yaml \
    -o Global.epochs=10 -o Global.eval_during_train=False \
    -o DataLoader.Train.sampler.batch_size=256 \
    -o DataLoader.Train.loader.num_workers=8 \
    -o AMP.level=O1 -o Global.save_interval=20 \
    -o Global.device=xpu \
    -o Global.use_dali=False \
    -o Global.to_static=False >> ${FLOG} 2>&1 # edit device type to npu or xpu
echo "End Time is: $(date +'%m/%d/%Y %T')" >> ${FLOG}
