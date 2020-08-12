#!/bin/bash
cur_dir=$(pwd)

wget --no-check-certificate -P ${cur_dir}/models/ https://paddlepaddle-inference-banchmark.bj.bcebos.com/MobileNet_SSD_infer_model.tar.gz