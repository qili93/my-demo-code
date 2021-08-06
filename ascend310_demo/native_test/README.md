### 代码复现说明：

1. 修改`shell/build.sh`脚本中的`HUAWEI_ASCEND_NPU_DDK_ROOT`，确认指向正确的NPU DDK目录，目录下必须寻在`acllib`, `atc`,`opp`三个目录。

   ```bash
   export HUAWEI_ASCEND_NPU_DDK_ROOT=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux_gcc4.8.5
   ```

2. 运行如下编译命令，生成程序可执行文件

   ```bash
   cd shell
   bash build.sh
   ```

3. 运行脚本，得到程序运行结果

   ```bash
   cd shell
   bash run_demo.sh
   ```

> 注意：IR算子的定义位于`shell/graph.h`文件的`GenGraph`函数中。