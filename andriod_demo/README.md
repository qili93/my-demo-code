# 如何修改Andriod Shell Demo

第一步：首先根据[源码编译环境准备](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html)准备好相应的编译开发环境

第二步：根据您的开发环境修改 `mnist_demo/shell`文件夹下的`build.sh`文件

- `ANDROID_NDK` 请指向您开发环境中的Andriod NKD路径

- `ANDROID_ABI` 请修改成您需要编译的ARMv8或者AMRv7

第三步：根据您的模型修改  `mnist_demo/shell`文件夹下的`model_test.cc`文件

- `INPUT_SHAPE` 请修改为您模型的输入大小，如果您的模型存在多个输入，则需要修改40-47行数据准备部分

第四步：把您的.nb模型拷贝到`assets/models`目录下

第五步：根据您的模型修改  `mnist_demo/shell`文件夹下的`model_test.cc`文件
- `ANDROID_NDK` 请指向您开发环境中的Andriod NKD路径

- `ANDROID_ABI` 请修改成您需要编译的ARMv8或者AMRv7

- `MODEL_NAME`修改为您模型.nb文件的名称

第六步：进行编译，在 `mnist_demo/shell`目录下直接执行如下命令即可

```bash
bash build.sh
```

第七步：连接手机到电脑，确认`adb devices -l`可以识别设备，在 `mnist_demo/shell`目录下直接执行如下命令即可运行demo

```bash
bash run_demo.sh
```

成功运行之后的输出如下：

```bash
results: 3u
Top0: 8 - 0.728158
Top1: 2 - 0.145191
Top2: 3 - 0.085126
```

> 注意：如果您修改了`ANDROID_ABI`，相应的inference_lite_lib下的编译库也需要修改成基于armv8/armv7编译的库

