import os
import csv
import git
import time
import subprocess
import paddle
# from IPython import get_ipython

kernel_path = "/workspace/PaddleCustomDevice/backends/npu/kernels"

# get latest commit and date
repo = git.Repo("/workspace/PaddleCustomDevice")
commit = repo.head.commit
commit_date = time.strftime("%Y%m%d", time.gmtime(commit.committed_date))
# add date to output file name
file_out = "kernel_author_" + commit_date + ".csv"

# 获取所有的_npu.cc文件
def get_filelist(dir):
    file_list = []
    for home, dirs, files in os.walk(dir):
        for filename in files:  
            if filename.endswith(".cc"):
                file_list.append(os.path.join(home, filename))
    return file_list
kernel_file_list = get_filelist(kernel_path)
print("kernel file number is : {}".format(len(kernel_file_list)))

def get_command(command):
    # print(command)
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    # print(out)
    return str(out.decode("utf-8"))

# 获取所有的NPU算子的名称和行数
kernel_support_dict = {}
for file_name in kernel_file_list:
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines)):
            line = lines[i]           
            if line.find("PD_REGISTER_PLUGIN_KERNEL(") != -1:
                if line.endswith("(\n"):
                    op_line = lines[i + 1]
                    op_name = op_line.strip().split(",")[0].strip()
                else:
                    op_line = line
                    op_name = op_line.strip().split("(")[1].split(",")[0].strip()
                author_name = get_command(f'cd {kernel_path} && git blame -p -L {i},{i} -- {file_name} | grep "author "')
                author_name = author_name.strip().split("author ")[1].strip()
                time_stamp =  get_command(f'cd {kernel_path} && git blame -L {i},{i} -- {file_name} | cut -d " " -f3')
                kernel_support_dict[op_name] = (author_name, time_stamp)
print("kernel number is : {}".format(len(kernel_support_dict)))

# SORT first and save to CSV
with open(file_out, 'w') as f:
    for key in sorted (kernel_support_dict.keys()):
        author_name, time_stamp = kernel_support_dict[key]
        f.write("{},{},{}".format(key, author_name, time_stamp))
