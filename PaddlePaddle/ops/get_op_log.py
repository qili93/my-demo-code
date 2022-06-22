# 使用说明：python getops.py train.log
import os
import sys
import csv

file_name = sys.argv[1]

# 获取每个文件的OP list，并保存成对应的csv文件
file_out = file_name.replace(".log", ".csv")
op_name_list = []
with open(file_name, 'r') as f:
    for line in f:
        if line.find("Op(") != -1:
            op_name = line.strip().split("Op(")[1].split(")")[0].strip()
            if not op_name in op_name_list:
                op_name_list.append(op_name)

# 先排序，再保存为 csv 文件
op_name_list.sort()
with open(file_out, 'w') as f:
    for item in op_name_list:
        f.write("%s\n" % item)
