#!/bin/bash

MODEL_DIR="../../assets/models/"
MODEL_PATH=$MODEL_DIR"simple_mnist"

if [ ! -f "$MODEL_PATH.nb" ]; then
  echo "$MODEL_PATH.nb not exist!!!"
  exit 1
fi

EXE_FILE="build-x86/mnist_lite"
GLOG_v=5 $EXE_FILE "$MODEL_PATH.nb" predict