#!/bin/bash

# git clone https://gitee.com/mindspore/transformer.git

# -------------- DataSet Preparetion -------------------
cd transformer/examples/preprocess/gptpreprocess

# download dataset for gpt3
# https://skylion007.github.io/OpenWebTextCorpus/

# data decompression
cd preprocess_gpt
tar xvJf openwebtext.tar.xz
cd openwebtext
xz -dk *

# data preprocess
cd preprocess_gpt
python pre_process.py \
--input_glob=./openwebtext/*
--dataset_type=openwebtext
--output_file=./output/openwebtext.mindrecord

# -------------- Single Card Train -------------------
# bash examples/pretrain/pretrain_gpt.sh DEVICE_ID EPOCH_SIZE DATA_DIR

python -m transformer.train \
--config='./transformer/configs/gpt/gpt_base.yaml' \
--epoch_size=10 \
--data_url=.\preprocess_gpt\output\ \
--optimizer="adam"  \
--seq_length=1024 \
--parallel_mode="stand_alone" \
--global_batch_size=4 \
--vocab_size=50257 \
--hidden_size=2048 \
--num_layers=24 \
--num_heads=16 \
--full_batch=False \
--device_target="GPU"
