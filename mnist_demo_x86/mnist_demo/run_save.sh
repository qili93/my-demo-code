#!/bin/bash

MODEL_DIR="../../assets/models"
MODEL_PATH="$MODEL_DIR/simple_mnist"

# delete model.nb before save
rm -rf "$MODEL_PATH.nb"
if [ $? -ne 0 ]; then
  echo "failed to delete $MODEL_PATH.nb"
else
  echo "succeed to delete $MODEL_PATH.nb"
fi

EXE_FILE="build/mnist_lite"
GLOG_v=5 $EXE_FILE $MODEL_PATH save