#!/bin/bash

# git clone https://github.com/PaddlePaddle/PaddleNLP.git -b develop

# install dependency
pip install regex sentencepiece tqdm visualdl pybind11 lac zstandard

# -------------- DataSet Preparetion -------------------

# download dataset for gpt3
wget https://mystic.the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar
tar -xvf openwebtext2.json.zst.tar -C  /datasets/openwebtext

# data preprocess
cd model_zoo/gpt/data_tools
python -u  create_pretraining_data.py \
    --model_name gpt2-en \
    --tokenizer_name GPTTokenizer \
    --data_format JSON \
    --input_path /datasets/openwebtext \
    --append_eos \
    --output_prefix gpt_openwebtext  \
    --workers 40 \
    --log_interval 10000

# -------------- Single Card Train -------------------
export CUDA_VISIBLE_DEVICES=0 

python run_pretrain.py \
    --model_type gpt \
    --model_name_or_path gpt2-en \
    --input_dir "gpt_openwebtext"\
    --output_dir "output"\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 500000\
    --save_steps 100000\
    --decay_steps 320000\
    --warmup_rate 0.01\
    --micro_batch_size 4\
    --device gpu
