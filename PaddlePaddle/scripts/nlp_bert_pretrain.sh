#!/bin/bash

# git clone https://github.com/PaddlePaddle/PaddleNLP.git -b develop


# -------------- DataSet Preparetion -------------------

# Here we use a specific commmit, the latest commit should also be fine.
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/LanguageModeling/BERT

# # Modified the parameters `--max_seq_length 512` to `--max_seq_length 384` at line 50 and
# # `--max_predictions_per_seq 80` to `--max_predictions_per_seq 56` at line 51.
# vim data/create_datasets_from_start.sh

# Build docker image
bash scripts/docker/build.sh

# Use NV's docker to download and generate hdf5 file. This may requires GPU available.
# You can Remove `--gpus $NV_VISIBLE_DEVICES` to avoid GPU requirements.
bash scripts/docker/launch.sh

# generate dataset with wiki_only
export http_proxy=http://172.19.57.45:3128
export https_proxy=http://172.19.57.45:3128
export ftp_proxy=http://172.19.57.45:3128
bash data/create_datasets_from_start.sh wiki_only

# rm data/wikipedia/* if download fail
rm -rf data/wikipedia/*
