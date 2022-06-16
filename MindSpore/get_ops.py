# 使用说明：python get_ops.py raw.txt
import os
import sys
import csv

file_name = sys.argv[1]
file_out = file_name.replace(".txt", ".csv")

# 获取每个文件的OP list，并保存成对应的csv文件
op_name_list = []
with open(file_name, 'r') as f:
  flag_seperate_line = False
  for line in f:
    op_name = None
    if flag_seperate_line:
      op_name = line.strip().split(",")[0].strip()
    if line.find("MS_REG_GPU_KERNEL") != -1:
      if line.endswith("("): 
        flag_seperate_line = True
        continue
      else:
        print(f"line={line}")
        op_name = line.strip().split("(")[1].split(",")[0].strip()
    if op_name and not op_name in op_name_list:
      op_name_list.append(op_name)

# 先排序，再保存为 csv 文件
op_name_list.sort()
with open(file_out, 'w') as f:
    for item in op_name_list:
        f.write("%s\n" % item)
