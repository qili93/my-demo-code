# PaddleLite使用华为昇腾NPU预测部署

Paddle Lite已支持华为昇腾310 NPU 在 X86服务器上进行预测部署。

原理是在线分析Paddle模型，将Paddle算子转成Huawei CANN的GE IR后，调用CANN的 IR/Builder/Runtime等API生成并执行模型预测。当前已经支持并验证多个模型在Atlas300推理加速卡上的精度和性能，后续会计划支持昇腾910相关系列产品。

## 参考示例演示

### 预先要求

请先根据华为昇腾Atlas300推理卡文档中的[CANN软件安装指南](https://support.huaweicloud.com/instg-A300_3000_3010/atlasdc_03_0002.html)准备开发和测试环境。

### 图像分类示例程序

**第一步：** 从[PaddleLite-X86_64-demo](https://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/PaddleLite-X86_64-demo.tar.gz)下载示例程序，解压后文件目录如下：

```shell
PaddleLite-X86_64-demo
├── README.md                                       # 示例程序README
├── image_classification_demo                       # 示例程序源码
│   ├── assets
│   │   ├── images
│   │   │   └── tabby_cat.jpg                       # 测试图片
│   │   ├── labels
│   │   │   └── synset_words.txt                    # 1000分类label文件
│   │   └── models
│   │       └── mobilenet_v1_fp32_224_fluid         # Paddle fluid non-combined格式的mobilenet_v1模型
│   │           ├── __model__                       # Paddle fluid模型组网文件，可用Netron查看网络结构
│   │           ├── conv1_bn_offset                 # Paddle fluid模型参数文件
│   │           ├── conv1_bn_mean
│   │           └── ...
│   └── shell
│       ├── build.sh                                # 编译脚本
│       ├── CMakeLists.txt                          # CMake脚本
│       ├── image_classification_demo.cc            # 示例程序源码
│       ├──run.sh                                   # 示例程序运行脚本
│       └── build
│           └── image_classification_demo           # 预编译的示例程序，基于Ubuntu
└── libs
    └── PaddleLite
        └── x86_64-linux_gcc7.3.0                   # 预编译的Lite C++预测库，基于Ubuntu
            ├── include
            │   ├── paddle_api.h                    # Paddle-Lite头文件
            │   └── ...
            └── lib                                 # 预编译的C++预测库
                ├── libiomp5.so                     # Intel OpenMP库
                ├── libmklml_intel.so               # Intel MKL库
                └── libpaddle_full_api_shared.so    # 预编译C++ FUll API库
```

**第二步：** 进入`PaddleLite-X86_64-demo/image_classification_demo/shell`，直接执行`bash run.sh`即可

```shell
cd PaddleLite-X86_64-demo/image_classification_demo/shell
bash run.sh

# 获得的预测结果如下
iter 0 cost: 2.175000 ms
iter 1 cost: 2.044000 ms
iter 2 cost: 2.151000 ms
iter 3 cost: 2.082000 ms
iter 4 cost: 2.147000 ms
iter 5 cost: 2.164000 ms
iter 6 cost: 2.169000 ms
iter 7 cost: 2.035000 ms
iter 8 cost: 2.153000 ms
iter 9 cost: 2.108000 ms
warmup: 5 repeat: 10, average: 2.122800 ms, max: 2.175000 ms, min: 2.035000 ms
results: 3
Top0  tabby, tabby cat - 0.529785
Top1  Egyptian cat - 0.419189
Top2  tiger cat - 0.044891
Preprocess time: 1.189000 ms
Prediction time: 2.122800 ms
Postprocess time: 0.309000 ms
```

如果需要重新编译示例程序，直接运行`bash build.sh`即可

```shell
cd PaddleLite-X86_64-demo/image_classification_demo/shell
bash build.sh
```

### 源码编译支持华为昇腾310的Paddle-Lite预测库

在准备好华为昇腾Atlas300开发和运行环境的前提下，调用Paddle-Lite 提供的源码编译脚本即可一键编译

```shell
# 1. 下载Paddle-Lite源码 并切换到Paddle-Lite目录下
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite

# (可选) 删除此目录，编译脚本会自动从国内CDN下载第三方库文件
# 此方法可节省从git repo同步第三方库代码的时间，缩短编译时间
# rm -rf third-party

# 2. 编译Paddle-Lite Huawwei Ascend NPU预测库
export HUAWEI_ASCEND_NPU_DDK_ROOT=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux_gcc7.3.0
bash lite/tools/build.sh --with_huawei_ascend_npu=ON \
     --huawei_ascend_npu_ddk_root=$HUAWEI_ASCEND_NPU_DDK_ROOT \
     x86
```

编译结果位于`Paddle-Lite/build.lite.huawei_ascend_npu/inference_lite_lib`，目录结构如下：

```shell
Paddle-Lite/build.lite.huawei_ascend_npu/inference_lite_lib
├── cxx                                               # C++ 预测库和头文件
│   ├── include                                       # C++ 头文件
│   │   ├── paddle_api.h
│   │   ├── paddle_lite_factory_helper.h
│   │   └── ...
│   └── lib                                           # C++ 预测库
│       ├── libpaddle_api_full_bundled.a              # C++ 静态库
│       └── libpaddle_full_api_shared.so              # C++ 动态库
└── third_party                                       
    └──mklml
        └── lib                                       # Intel MKL库
            ├── libiomp5.so
            ├── libmklml_gnu.so
            └── libmklml_intel.so
```

将相应的头文件和库文件拷贝到`PaddleLite-X86_64-demo/lib/`目录下，即可更新支持Huawei Ascend NPU的Paddle-Lite库。

## 其它说明

- 百度Paddle-Lite正在持续增加能够适配CANN GE IR的Paddle算子bridge/converter，以便适配更多Paddle模型。
- 如需更进一步的了解相关产品的信息，请联系百度李琦liqi27@baidu.com。

