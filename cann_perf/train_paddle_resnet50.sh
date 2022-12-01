#!/bin/bash
set -ex

DEVICE_TARGET=${1:-ascend} # cpu, gpu, npu, ascend
DEVICE_ID=${2:-0} # device id

echo "======== Eager Mode ========"
python3 paddle_resnet50_eager.py --device=${DEVICE_TARGET} --ids=${DEVICE_ID} --amp=O0 > paddle_resnet50_eager_amp_o0.log 2>&1
sleep 10s
python3 paddle_resnet50_eager.py --device=${DEVICE_TARGET} --ids=${DEVICE_ID} --amp=O1 > paddle_resnet50_eager_amp_o1.log 2>&1
sleep 10s


echo "======== Dy2Static Mode ========"

python3 paddle_resnet50_eager.py --device=${DEVICE_TARGET} --ids=${DEVICE_ID} --amp=O0 --to_static > paddle_resnet50_eager2static_amp_o0.log 2>&1
sleep 10s
python3 paddle_resnet50_eager.py --device=${DEVICE_TARGET} --ids=${DEVICE_ID} --amp=O1 --to_static > paddle_resnet50_eager2static_amp_o1.log 2>&1
sleep 10s

echo "======== Static Mode ========"
python3 paddle_resnet50_graph.py --device=${DEVICE_TARGET} --ids=${DEVICE_ID} --amp=O0 > paddle_resnet50_graph_amp_o0.log 2>&1
sleep 10s
python3 paddle_resnet50_graph.py --device=${DEVICE_TARGET} --ids=${DEVICE_ID} --amp=O1 > paddle_resnet50_graph_amp_o1.log 2>&1
