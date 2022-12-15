import os
import sys
import csv

# 输入文件
op_file_1 = sys.argv[1]
op_file_2 = sys.argv[2]

# 输出文件
ouptut_comm = "ouptut_comm.csv"
output_diff1 = "output_diff1.csv"
output_diff2 = "output_diff2.csv"

# 获取模型算子列表
op_list_1 = []
with open(op_file_1) as f:
    for row in csv.reader(f):
        op_list_1.append(row[0])
print(f"op number of {op_file_1} is {len(op_list_1)}")

# 获取模型算子列表
op_list_2 = []
with open(op_file_2) as f:
    for row in csv.reader(f):
        op_list_2.append(row[0])
print(f"op number of {op_file_2} is {len(op_list_2)}")

# 获取 common 算子
op_comm_list = []
op_diff_list_1 = []
for op_name in op_list_1:
    if op_name in op_list_2:
        op_comm_list.append(op_name)
    else:
        op_diff_list_1.append(op_name)
print(f"comm op number is {len(op_comm_list)}")
print(f"diff of {op_file_1} is {len(op_diff_list_1)}")

# 获取 common 算子
op_comm_list = []
op_diff_list_2 = []
for op_name in op_list_2:
    if op_name in op_list_1 and op_name not in op_comm_list:
        op_comm_list.append(op_name)
    else:
        op_diff_list_2.append(op_name)
print(f"comm op number is {len(op_comm_list)}")
print(f"diff of {op_file_2} is {len(op_diff_list_2)}")

# 先排序
op_comm_list.sort()
op_diff_list_1.sort()
op_diff_list_2.sort()

# 写入CSV文件
with open(ouptut_comm, 'w') as f:
    for item in op_comm_list:
        f.write("%s\n" % item)
with open(output_diff1, 'w') as f:
    for item in op_diff_list_1:
        f.write("%s\n" % item)
with open(output_diff2, 'w') as f:
    for item in op_diff_list_2:
        f.write("%s\n" % item)