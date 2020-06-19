#!/bin/bash

MODEL_DIR="../../assets/models"
MODEL_PATH="$MODEL_DIR/simple_mnist.nb"

if [ ! -f "$MODEL_PATH" ]; then
  echo "$MODEL_PATH NOT exist!!!"
  exit 1
fi

EXE_FILE="build/mnist_lite"
GLOG_v=5 $EXE_FILE $MODEL_PATH predict