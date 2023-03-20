import os
import sys
import csv

import paddle
import numpy as np
from datetime import date

def kernel_filter(all_kernel_list):
  valid_kernel_list = []
  # phi_ops_all = paddle.fluid.core.get_all_op_names("phi")
  for op_name in all_kernel_list:
      if op_name.endswith("_sparse_grad"):
          continue
      if op_name.endswith("_grad_grad"):
          continue
      if op_name.endswith("_double_grad"):
          continue
      if op_name.endswith("_triple_grad"):
          continue
      if op_name.startswith("sparse_"):
          continue
      if op_name.startswith("graph_send_"):
          continue
      if op_name.startswith("fused_"):
          continue
      if op_name.startswith("fft_"):
          continue
      if op_name.startswith("feed"):
          continue
      if op_name.startswith("fetch"):
          continue
      valid_kernel_list.append(op_name)
  return valid_kernel_list

def get_kernel_by_place(target_place):
  op_support_list = []
  kernel_dict = paddle.fluid.core._get_all_register_op_kernels('phi')
  for op_name in kernel_dict:
      kernel_list = kernel_dict[op_name]
      for item in kernel_list:
          # print(f"op_name = {op_name}, item = {item}")
          if item.find(target_place) != -1 and op_name not in op_support_list:
              op_support_list.append(op_name)
  return op_support_list

def get_target_place():
    # get target place and output file name
    # place = "dcu" if paddle.is_compiled_with_rocm() else sys.argv[1]
    # output_file = f"{place}_ops.csv"
    target_place = f"place[Place({sys.argv[1]}:0)]"
    if sys.argv[1] == 'cpu':
        target_place = f"place[Place({sys.argv[1]})]"
    return target_place

if __name__ == '__main__':
    target_place = get_target_place()

    # for debug - phi ops filter
    phi_ops_all = paddle.fluid.core.get_all_op_names("phi")
    phi_ops_all.sort()
    with open("phi_ops_all.csv", 'w') as f:
        for item in phi_ops_all:
            f.write("%s\n" % item)
    phi_ops_cln = kernel_filter(phi_ops_all)
    phi_ops_cln.sort()
    with open("phi_ops_cln.csv", 'w') as f:
        for item in phi_ops_cln:
            f.write("%s\n" % item)

    # for debug - dev ops filer
    dev_ops_all = get_kernel_by_place(target_place)
    dev_ops_all.sort()
    with open("dev_ops_all.csv", 'w') as f:
        for item in dev_ops_all:
            f.write("%s\n" % item)
    dev_ops_cln = kernel_filter(dev_ops_all)
    dev_ops_cln.sort()
    with open("dev_ops_cln.csv", 'w') as f:
        for item in dev_ops_cln:
            f.write("%s\n" % item)

    # get clean kernels from phi ops
    # phi_ops_cln = kernel_filter(paddle.fluid.core.get_all_op_names("phi"))
    # dev_ops_cln = kernel_filter(get_kernel_by_place(target_place))

    # kernel_list = []
    # for op_name in dev_ops_cln:
    #     if op_name in phi_ops_cln:
    #       kernel_list.append(op_name)
    # kernel_list.sort()

    kernel_list = np.intersect1d(dev_ops_cln, phi_ops_cln)

    # 写入 csv 文件
    place = "dcu" if paddle.is_compiled_with_rocm() else sys.argv[1]
    today = date.today().strftime("%Y-%m-%d")
    output_file = f"{place}_ops_" + today + ".csv"
    with open(output_file, 'w') as f:
        for item in kernel_list:
            f.write("%s\n" % item)
