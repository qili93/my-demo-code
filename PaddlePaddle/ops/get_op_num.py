# Usage:
# 1. pip install gitpython
# 2. Install paddlepaddle-gpu or other whls
# 3. python get_ops.py cpu/gpu/npu/xpu/mlu

import os
import sys
import csv
import git
import time
import paddle

# get commit date by paddle.version.commit
paddle_commit = paddle.version.commit
repo = git.Repo("/workspace/Paddle")
commit = list(repo.iter_commits(paddle_commit, max_count=1))[0]
commit_date = time.strftime("%Y%m%d", time.gmtime(commit.committed_date))

# get target place and output file name
output_file = f"{sys.argv[1]}_ops_" + commit_date + ".csv"
# target_place = f"place[Place({sys.argv[1]}:0)]"
# if sys.argv[1] == 'cpu':
#    target_place = f"place[Place({sys.argv[1]})]"
target_place = f"place[{sys.argv[1].upper()}Place(0)]"
if sys.argv[1] == 'cpu':
   target_place = f"place[{sys.argv[1].upper()}Place]"

print(f"---------- Target: {target_place}, Output: {output_file} ----------")

# get op list by _get_all_register_op_kernels
op_support_list = []
kernel_dict = paddle.fluid.core._get_all_register_op_kernels()
for op_name in kernel_dict:
    kernel_list = kernel_dict[op_name]
    for item in kernel_list:
        # print(f"op_name = {op_name}, item = {item}")
        if item.find(target_place) != -1 and op_name not in op_support_list:
            op_support_list.append(op_name)

# 先排序, 写入 csv 文件
op_support_list.sort()
with open(output_file, 'w') as f:
    for item in op_support_list:
        f.write("%s\n" % item)
