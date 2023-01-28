#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import numpy as np

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--op',
      type=str,
      help="Op name to check")
  return parser.parse_args()

def get_filelist(op_type):
  file_list = []
  for root, dirs, files in os.walk("."):
      for filename in files:
          if (filename.find(op_type) != -1):
              file_list.append(os.path.join(root, filename))
  return file_list

def get_data_value(file_name):
  dtype = np.float32 # default
  with open(file_name, 'r') as f:
      for line in f:
          if line.find("TensorDesc") != -1:
              dtype_str = line.strip().split("data_type = ")[1].split(",")[0]
              if (dtype_str == 'ACL_FLOAT16'): dtype = np.float16
          if line.find("TensorData:") != -1:
              data_str = line.strip().split("[")[1].split("]")[0]
              data = np.fromstring(data_str, dtype=dtype, sep=', ')
  return data

def main(args):
  for file_name in get_filelist(args.op):
    print(f"Checking: {file_name}")
    data_array = get_data_value(file_name)
    if np.isnan(data_array).any():
        print(f"-- {file_name} contains Nan")
    if np.isinf(data_array).any():
        print(f"-- {file_name} contains Inf")

if __name__ == '__main__':
    args = parse_args()
    print('---------------  Running Arguments ---------------')
    print(args)
    print('--------------------------------------------------')

    main(args)
