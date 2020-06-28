### Demo运行说明：

1. 修改每个demo文件夹下的shell/build.sh文件中的`ANDROID_NDK`路径，指向本机的NDK路径

   ```bash
   ANDROID_NDK=/Users/<username>/Library/android-ndk-r17c # Mac OS
   ```

2. 运行如下编译命令，生成模型可执行文件

   ```bash
   bash build.sh --arm_abi=armv8 --with_log=ON full_build
   ```

3. 运行脚本，得到模型运行结果

   ```bash
   bash run_demo.sh --arm_abi=armv8 --with_log=ON full_demo
   ```

> 运行脚本需注意：build.sh/run_demo.sh的输入参数必须一致；在with_log=OFF tiny_build/tiny_demo下性能损耗最小。