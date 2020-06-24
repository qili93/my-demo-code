### Demo运行说明：

1. 修改每个demo文件夹下的shell/build.sh文件中的`ANDROID_NDK`路径，指向本机的NDK路径

   ```bash
   ANDROID_NDK=/Users/<username>/Library/android-ndk-r17c # Mac OS
   ```

2. 运行如下编译命令，生成模型可执行文件

   ```bash
   bash build.sh
   ```

3. 运行脚本，得到模型运行结果

   ```bash
   bash run.sh
   ```


