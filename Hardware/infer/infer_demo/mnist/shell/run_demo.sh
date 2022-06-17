#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. compile
bash ${work_path}/compile.sh

# 2. run
export CUSTOM_DEVICE_ROOT=/workspace/PaddleCustomDevice/backends/custom_cpu/build
./build/mnist_test --model_file ../assets/mnist.pdmodel --params_file ../assets/mnist.pdiparams
