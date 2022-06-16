# Usage:
# 1. pip install tf-nightly
# 2. python get_ops.py GPU/CPU

import os
import sys
import csv
import tensorflow as tf
from tensorflow.python.framework import kernels

device_type = f"{sys.argv[1]}" # CPU or GPU
output_file = f"{sys.argv[1]}_ops_support.csv"

op_support_list = []
reg_kernels = kernels.get_all_registered_kernels()
for kernel in reg_kernels.kernel:
  if kernel.device_type == device_type and kernel.op not in op_support_list:
    op_support_list.append(kernel.op)

op_support_list.sort()
with open(output_file, 'w') as f:
    for item in op_support_list:
        f.write("%s\n" % item)

