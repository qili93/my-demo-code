#!/bin/bash
cur_dir=$(pwd)

wget --no-check-certificate -P ${cur_dir}/models/ https://paddlelite-demo.bj.bcebos.com/models/ernie_fp32_fluid.tar.gz