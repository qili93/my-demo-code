#!/bin/bash
cur_dir=$(pwd)

wget --no-check-certificate -P ${cur_dir}/models/ https://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz