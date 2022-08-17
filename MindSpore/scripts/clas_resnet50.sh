#!/bin/bash

# git clone https://gitee.com/mindspore/models.git
# prepare dataset of ILSVRC2012

export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# apply clas_resnet50.diff if needed to change run mode and disable log for pynative

FLOG="mindspore-resnet50-imagenet-ascend-card1-bs256-graph.log"
echo "Start Time is: $(date +'%m/%d/%Y %T')" > ${FLOG}
python train.py  --data_path=/datasets/ILSVRC2012/train \
       --config_path=config/resnet50_imagenet2012_config.yaml \
       --output_path './output'  >> ${FLOG} 2>&1
echo "End Time is: $(date +'%m/%d/%Y %T')" >> ${FLOG}

