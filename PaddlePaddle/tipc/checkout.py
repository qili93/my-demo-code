import subprocess
cmd_output = subprocess.getoutput("find ./ -name 'results_python.log'")
result_files = cmd_output.split("\n")

# =============== get data from log ===============
def check_status(file_name):
    success_flag = False
    with open(file_name, 'r') as f:
        for line in f:
            if line.find("Run failed with command") != -1:
                return False
            if success_flag == False and line.find("Run successfully with command") != -1:
                success_flag = True
    return success_flag

result_dict = {}
for file in result_files:
    model_name = file.split("/")[1].strip()
    if check_status(file) == True:
        result_dict[model_name] = 1
    else:
        result_dict[model_name] = 0
print(result_dict)

import csv

with open('check_status.csv', 'w') as f:
    fieldnames = ['model_name', 'status']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for key in sorted(result_dict):
        w.writerow({'model_name': key, 'status': result_dict[key]})

