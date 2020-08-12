#!/bin/bash
cur_dir=$(pwd)

wget --no-check-certificate -P ${cur_dir}/models/ https://paddlepaddle-inference-banchmark.bj.bcebos.com/ResNet50_inference.tar
wget --no-check-certificate -P ${cur_dir}/models/ https://paddlepaddle-inference-banchmark.bj.bcebos.com/ResNet101_inference.tar