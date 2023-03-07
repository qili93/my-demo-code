import os
import os.path

ext = [".cmake", ".cc", ".h", "cpp", ".hpp", ".py", ".txt", ".in"]

souce_list = []

for root, dirs, files in os.walk("./backends/npu/"):
  for filename in files:
    file_path = os.path.join(root, filename)
    if file_path.find("./.git/") != -1:
      continue
    if os.path.islink(file_path):
      print(f"skipping file: {file_path} -- link")
      continue
    if not file_path.endswith(tuple(ext)):
      print(f"skipping file: {file_path} -- ext not match")
      continue
    souce_list.append(file_path)

print("---------------------------")
print("---------------------------")
print("---------------------------")
print(souce_list)


with open('output_file_npu.txt', 'w') as f_out:
  for file_path in souce_list:
    with open(file_path, 'r') as f_in:
      f_out.write(f_in.read())

