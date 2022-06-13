import os
import sys
import csv
import paddle

# Usage:
# 1. install paddlepaddle whl package first
# 2. python get_ops.py gpu/npu/ipu/xpu

target_place = f"place[Place({sys.argv[1]}:0)]"
if sys.argv[1] == 'cpu':
   target_place = f"place[Place({sys.argv[1]})]"
output_file = f"{sys.argv[1]}_ops_support.csv"
print(f"---------- OP Number for Target: {target_place} ----------")

op_support_list = []
kernel_dict = paddle.fluid.core._get_all_register_op_kernels()
for op_name in kernel_dict:
    kernel_list = kernel_dict[op_name]
    for item in kernel_list:
        # print(f"op_name = {op_name}, item = {item}")
        if item.find(target_place) != -1:
            if op_name not in op_support_list:
                op_support_list.append(op_name)

# 先排序
op_support_list.sort()
# 写入 csv 文件
with open(output_file, 'w') as f:
    for item in op_support_list:
        f.write("%s\n" % item)



