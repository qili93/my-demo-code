import os
import csv

# 输入文件
npu_support_file = "ascend_ops_20220901.csv"
npu_require_file = "npu_ops_20220901.csv"

# 输出文件
migrate_support = "npu_ops_support.csv"
migrate_missing = "npu_ops_missing.csv"

# 获取模型算子列表
npu_require_list = []
with open(npu_require_file) as f:
    for row in csv.reader(f):
        npu_require_list.append(row[0])
print(len(npu_require_list))

# 获取已经支持的NPU算子
npu_support_list = []
with open(npu_support_file) as f:
    for row in csv.reader(f):
        npu_support_list.append(row[0])
print(len(npu_support_list))

# 输出还需开发的NPU算子
npu_support_list_new = []
npu_missing_list_new = []
for op_name in npu_require_list:
    if op_name in npu_support_list:
        npu_support_list_new.append(op_name)
    else: # if not op_name in npu_support_list:
        npu_missing_list_new.append(op_name)
print("Support OP: ", len(npu_support_list_new))
print("Develop OP: ", len(npu_missing_list_new))

# 输出其他算子
npu_other_list = []
for op_name in npu_support_list:
    if op_name not in npu_require_list:
      npu_other_list.append(op_name)
print("========= Other OP ========= ")
print(npu_other_list)

# 先排序
npu_support_list_new.sort()
npu_missing_list_new.sort()
# 写入CSV文件
with open(migrate_support, 'w') as f:
    for item in npu_support_list_new:
        f.write("%s\n" % item)
# 写入CSV文件
with open(migrate_missing, 'w') as f:
    for item in npu_missing_list_new:
        f.write("%s\n" % item)