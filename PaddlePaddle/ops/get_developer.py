import os
import csv
import git
import time
import subprocess
import paddle
# from IPython import get_ipython

repo_dir = "/workspace/PaddleCustomDevice/backends/npu/kernels"


# 进入Paddle目录
# get_ipython().run_line_magic('cd', '{repo_dir}')
# get commit date by paddle.version.commit
paddle_commit = paddle.version.commit
repo = git.Repo("/workspace/Paddle")
commit = list(repo.iter_commits(paddle_commit, max_count=1))[0]
commit_date = time.strftime("%Y%m%d", time.gmtime(commit.committed_date))

file_out = "kernel_develop_" + commit_date + ".csv"

# 获取所有的_npu.cc文件
def get_filelist(dir):
    file_list = []
    for home, dirs, files in os.walk(dir):
        for filename in files:  
            if filename.endswith(".cc"):
                file_list.append(os.path.join(home, filename))
    return file_list
kernel_file_list = get_filelist(repo_dir)
print("kernel file number is : {}".format(len(kernel_file_list)))

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
                p = subprocess.Popen(f'cd {repo_dir} && git blame -p -L {i},{i} -- {file_name} | grep "author "', stdout=subprocess.PIPE, shell=True)
                out, err = p.communicate()
                author_name = ''.join(str(out.decode("utf-8"))).strip().split("author ")[1].strip()
                kernel_support_dict[op_name] = author_name
print("kernel number is : {}".format(len(kernel_support_dict)))

# SORT first and save to CSV
with open(file_out, 'w') as f:
    for key in sorted (kernel_support_dict.keys()):
        f.write("{},{}\n".format(key, kernel_support_dict[key]))

for key in sorted (kernel_support_dict.keys()):
    print("{},{}".format(key, kernel_support_dict[key]))