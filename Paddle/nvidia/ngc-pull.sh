#!/bin/bash

FLOG="ngc-speed-$(date +'%Y-%m-%d-%H-%M-%S').log"
echo "Start Time is: $(date +'%m/%d/%Y %T')" > ${FLOG}
docker pull nvcr.io/nvidia/paddlepaddle:22.05-py3 >> ${FLOG} 2>&1
echo "End Time is: $(date +'%m/%d/%Y %T')" >> ${FLOG}

