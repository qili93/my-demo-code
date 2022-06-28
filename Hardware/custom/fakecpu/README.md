# 飞桨自定义接入硬件后端(FakeCPU)

请参考以下步骤进行硬件后端(FakeCPU)的编译安装与验证，当前仅支持通过源码编译的方式安装。

## 一、环境与源码准备

请根据[编译依赖表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html)准备符合版本要求的依赖库，推荐使用飞桨官方镜像。

```bash
# 1) 拉取镜像并启动容器，该镜像基于 Ubuntu 18.04 操作系统构建
docker pull registry.baidubce.com/device/paddle-cpu:ubuntu18-x86_64-gcc82
docker run -it --name paddle-dev -v `pwd`:/workspace \
     --network=host --shm-size=128G --workdir=/workspace \
     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
     registry.baidubce.com/device/paddle-cpu:ubuntu18-x86_64-gcc82 /bin/bash

# 2) 克隆代码
git clone --recursive https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice

# 3) 同步源码，请执行以下命令以保证 checkout 最新的 Paddle 源码
git submodule sync
git submodule update --remote --init --recursive
```

## 二、Python 安装与验证

PaddlePaddle 支持 Python 训练和推理，请参考以下步骤进行 FakeCPU 的编译安装并运行 Python 训练和推理示例。

**第一步**：准备编译依赖环境，即飞桨 Python WHL 安装包

```bash
# 可以直接安装 CPU 版本
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
```

**第二步**：源码编译得到 paddle-custom-cpu whl 安装包

```bash
# 1) 进入硬件后端(昇腾NPU)目录
cd PaddleCustomDevice/backends/fake_cpu

# 2) 修改 compile.sh
ON_INFER=OFF # 设置 ON_INFER 为 OFF

# 3) 执行编译
bash ./compile.sh

# 4) 检查 build 目录下正确生成 whl 包
build/dist/
└── paddle_fake_cpu-0.0.0-cp37-cp37m-linux_x86_64.whl

# 5) 安装生成的 paddle-custom-npu whl 包
pip3 install -U build/dist/paddle_fake_cpu-0.0.0-cp37-cp37m-linux_x86_64.whl

# 6) 检查 paddle-plugin 目录下存在动态库
SITE_PACKAGE_PATH=`python3 -c 'import site; print(site.getsitepackages()[0])'`
tree $SITE_PACKAGE_PATH/paddle-plugins/
# 预期获得如下输出结果
/opt/conda/lib/python3.7/site-packages/paddle-plugins/
└── libpaddle-fake-cpu.so

# 7) 列出可用硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 预期获得如下输出结果
['custom_cpu']
```

**第三步**：Python 训练和推理功能验证

```bash
# 1) 进入 demo 目录
cd PaddleCustomDevice/backends/custom_cpu/demo/python

# 2) 训练功能验证
python mnist_train.py
# 预期获得如下输出结果
I0620 14:55:47.834406 11853 init.cc:265] CustomDevice: custom_cpu, visible devices count: 1
Loss at epoch 0 step 0: [2.3429298]
Loss at epoch 0 step 100: [2.058938]
... ...
Loss at epoch 0 step 900: [1.8213819]
Loss at epoch 0 , Test avg_loss is: 1.8066727159879146, acc is: 0.6594551282051282
Saved inference model to mnist
# 且训练完成之后当前目录下会生成训练后保存的推理模型
├── mnist.pdiparams
└── mnist.pdmodel

# 3) 推理功能验证
python mnist_infer.py
# 预期获得如下输出结果
Inference result of infer_3.png is:  3
```

## 三、C++ 安装与验证

PaddlePaddle 支持 C++ 推理，请参考以下步骤进行 Custom CPU 的编译安装并运行 C++ 训练和推理示例。

**第一步**：准备编译依赖环境，即 Paddle Inference C++ Lib 库

飞桨官方发布的 Paddle Inference C++ 预编译库默认无法支持自定义硬件后端注册，请参考以下步骤通过源码编译得到飞桨 C++ 推理库。

```bash
# 1) 进入 PaddlePaddle 源码目录，并拉取飞桨最新代码
cd PaddleCustomDevice/Paddle
git checkout develop && git pull

# 2) 准备编译目录
mkdir build && cd build

# 3) CMake 命令
cmake .. -DPY_VERSION=3.7 -DON_INFER=ON -DWITH_TESTING=OFF -DWITH_CUSTOM_DEVICE=ON

# 4) 执行编译
make -j$(nproc)

# 5) 编译完成之后 build 目录下的 paddle_inference_install_dir 目录即为 C++ 推理库，目录结构如下
PaddleCustomDevice/Paddle/build/paddle_inference_install_dir/
├── CMakeCache.txt
├── paddle
│   ├── include                                    # C++ 预测库头文件目录
│   │   ├── crypto
│   │   ├── experimental
│   │   ├── internal
│   │   ├── paddle -> experimental                 # 自定义硬件后端依赖的头文件目录
│   │   ├── paddle_analysis_config.h
│   │   ├── paddle_api.h
│   │   ├── paddle_infer_contrib.h
│   │   ├── paddle_infer_declare.h
│   │   ├── paddle_inference_api.h                 # C++ 预测库头文件
│   │   ├── paddle_mkldnn_quantizer_config.h
│   │   ├── paddle_pass_builder.h
│   │   └── paddle_tensor.h
│   └── lib
│       ├── libpaddle_inference.a                  # C++ 静态预测库文件
│       └── libpaddle_inference.so                 # C++ 动态态预测库文件
├── third_party
│   ├── install                                    # 第三方链接库和头文件
│   │   ├── cryptopp
│   │   ├── gflags
│   │   ├── glog
│   │   ├── mkldnn
│   │   ├── mklml
│   │   ├── protobuf
│   │   ├── utf8proc
│   │   └── xxhash
│   └── threadpool
│       └── ThreadPool.h
└── version.txt                                    # 预测库版本信息
```

**第二步**：源码编译得到 paddle-custom-cpu 动态链接库

```bash
# 1) 进入硬件后端(昇腾NPU)目录
cd PaddleCustomDevice/backends/custom_cpu

# 2) 修改 compile.sh
ON_INFER=ON # 需设置 ON_INFER 为 ON
LIB_DIR=${work_path}/../../Paddle/build/paddle_inference_install_dir # 设置为上一步生成的推理库路径

# 3) 执行编译
bash ./compile.sh

# 4) 检查 build 目录下正确生成动态库
build/libpaddle-custom-cpu.so
```

**第三步**：C++ 推理功能验证

```bash
# 1) 进入 demo 目录
cd PaddleCustomDevice/backends/custom_cpu/demo/c++

# 2) 修改 compile.sh，需根据 C++ 预测库的 version.txt 信息对以下的几处内容进行修改
WITH_MKL=ON
WITH_GPU=OFF

# 3) 执行编译，编译完成之后在 build 下生成 mnist_test 可执行文件
bash compile.sh

# 4) 指向 libpaddle-custom-cpu.so 所在目录
export CUSTOM_DEVICE_ROOT=/path/to/PaddleCustomDevice/backends/custom_cpu/build
export CUSTOM_DEVICE_ROOT=/workspace/PaddleCustomDevice/backends/custom_cpu/build
# 5) 运行 C++ 预测程序
./build/mnist_test  --model_file ../python/mnist.pdmodel --params_file ../python/mnist.pdiparams
# 预期获得如下输出结果
Inference result of infer_3.png is:  3
```
