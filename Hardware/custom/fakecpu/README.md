# 飞桨自定义接入硬件后端(FakeNPU)

简体中文 | [English](./README.md)

## 环境与源码准备

请根据[编译依赖表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html)准备符合版本要求的依赖库，推荐使用飞桨官方镜像。

**第一步**： 从飞桨镜像库拉取编译镜像并启动容器，该镜像基于 Ubuntu 18.04 操作系统构建

```bash
# 拉取镜像
docker pull registry.baidubce.com/device/paddle-cpu:ubuntu18-x86_64-gcc82
# 启动容器
docker run -it --name paddle-dev -v `pwd`:/workspace \
     --network=host --shm-size=128G --workdir=/workspace \
     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
     registry.baidubce.com/device/paddle-cpu:ubuntu18-x86_64-gcc82 /bin/bash
```

**第二步**：下载 PaddleCustomDevice 源码

```bash
# 克隆代码
git clone --recursive https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice

# 请执行以下命令，以保证 checkout 最新的 Paddle 源码
git submodule sync
git submodule update --remote --init --recursive
```

## 二、Python 安装与验证

**第一步**：准备编译依赖环境，即飞桨 Python WHL 安装包

```bash
# 可以直接安装 CPU 版本
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
```

**第二步**：源码编译得到 paddle-custom-npu whl 安装包

```bash
# 修改 compile.sh
ON_INFER=OFF # 设置 ON_INFER 为 OFF

# 执行编译
bash ./compile.sh

# 检查 build 目录下正确生成 whl 包
build/dist/
└── paddle_custom_npu-0.0.0-cp37-cp37m-linux_x86_64.whl

# 安装生成的 paddle-custom-npu whl 包
pip3 install -U build/dist/paddle_custom_npu-0.0.0-cp37-cp37m-linux_x86_64.whl

# 检查 paddle-plugin 目录下存在动态库
SITE_PACKAGE_PATH=`python3 -c 'import site; print(site.getsitepackages()[0])'`
tree $SITE_PACKAGE_PATH/paddle-plugins/
# 预期获得如下输出结果
/opt/conda/lib/python3.7/site-packages/paddle-plugins/
└── libpaddle-custom-npu.so
```

**第三步**：训练功能验证

```bash
# 列出可用硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 预期获得如下输出结果
['ascend']

# 运行简单模型
python tests/test_MNIST_model.py
# 预期获得如下输出结果
... ...
Epoch 0 step 0, Loss = [2.3313463], Accuracy = 0.046875
Epoch 0 step 100, Loss = [1.9624571], Accuracy = 0.484375
Epoch 0 step 200, Loss = [2.002725], Accuracy = 0.453125
Epoch 0 step 300, Loss = [1.912869], Accuracy = 0.546875
Epoch 0 step 400, Loss = [1.9169667], Accuracy = 0.5625
Epoch 0 step 500, Loss = [1.9007692], Accuracy = 0.5625
Epoch 0 step 600, Loss = [1.8512673], Accuracy = 0.625
Epoch 0 step 700, Loss = [1.8759218], Accuracy = 0.59375
Epoch 0 step 800, Loss = [1.8942316], Accuracy = 0.5625
Epoch 0 step 900, Loss = [1.8966292], Accuracy = 0.5625
```

## 三、推理安装与验证

**第一步**：准备编译依赖环境，

```bash
# 可以直接安装 CPU 版本
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
```

**第二步**：源码编译得到 paddle-custom-npu whl 安装包

```bash
# 进入硬件后端(昇腾NPU)目录
cd backends/npu

# 修改 compile.sh
ON_INFER=OFF # 训练下设置 ON_INFER 为 OFF

# 执行编译
bash ./compile.sh

# 检查 build 目录下正确生成 whl 包
build/dist/
└── paddle_custom_npu-0.0.0-cp37-cp37m-linux_x86_64.whl

# 安装生成的 paddle-custom-npu whl 包
pip3 install -U build/dist/paddle_custom_npu-0.0.0-cp37-cp37m-linux_x86_64.whl

# 检查 paddle-plugin 目录下存在动态库
SITE_PACKAGE_PATH=`python3 -c 'import site; print(site.getsitepackages()[0])'`
tree $SITE_PACKAGE_PATH/paddle-plugins/
# 预期获得如下输出结果
/opt/conda/lib/python3.7/site-packages/paddle-plugins/
└── libpaddle-custom-npu.so
```

**第三步**：训练功能验证

```bash
# 列出可用硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 预期获得如下输出结果
['ascend']

# 运行简单模型
python tests/test_MNIST_model.py
# 预期获得如下输出结果
... ...
Epoch 0 step 0, Loss = [2.3313463], Accuracy = 0.046875
Epoch 0 step 100, Loss = [1.9624571], Accuracy = 0.484375
Epoch 0 step 200, Loss = [2.002725], Accuracy = 0.453125
Epoch 0 step 300, Loss = [1.912869], Accuracy = 0.546875
Epoch 0 step 400, Loss = [1.9169667], Accuracy = 0.5625
Epoch 0 step 500, Loss = [1.9007692], Accuracy = 0.5625
Epoch 0 step 600, Loss = [1.8512673], Accuracy = 0.625
Epoch 0 step 700, Loss = [1.8759218], Accuracy = 0.59375
Epoch 0 step 800, Loss = [1.8942316], Accuracy = 0.5625
Epoch 0 step 900, Loss = [1.8966292], Accuracy = 0.5625
```



## 二、编译安装

```bash
# 进入硬件后端(昇腾NPU)目录
cd backends/npu

# 编译之前需要先保证环境下装有Paddle WHL包，可以直接安装CPU版本
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html

# 创建编译目录并编译
mkdir build && cd build

# X86_64环境编译
cmake ..
make -j8

# Aarch64环境编译
cmake .. -DWITH_ARM=ON
make TARGET=ARMV8 -j8

# 编译产出在dist路径下，使用pip安装
pip install dist/paddle_custom_npu*.whl
```

## 三、功能验证

```bash
# 列出可用硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 期待输出以下结果
['ascend']

# 运行简单模型
python ../tests/test_MNIST_model.py
# 期待输出以下类似结果
... ...
Epoch 0 step 0, Loss = [2.3313463], Accuracy = 0.046875
Epoch 0 step 100, Loss = [1.9624571], Accuracy = 0.484375
Epoch 0 step 200, Loss = [2.002725], Accuracy = 0.453125
Epoch 0 step 300, Loss = [1.912869], Accuracy = 0.546875
Epoch 0 step 400, Loss = [1.9169667], Accuracy = 0.5625
Epoch 0 step 500, Loss = [1.9007692], Accuracy = 0.5625
Epoch 0 step 600, Loss = [1.8512673], Accuracy = 0.625
Epoch 0 step 700, Loss = [1.8759218], Accuracy = 0.59375
Epoch 0 step 800, Loss = [1.8942316], Accuracy = 0.5625
Epoch 0 step 900, Loss = [1.8966292], Accuracy = 0.5625
```

## 四、使用PaddleInference

重新编译插件

```bash
# 编译PaddleInference
git clone https://github.com/PaddlePaddle/Paddle.git
git clone https://github.com/ronny1996/Paddle-Inference-Demo.git

mkdir -p Paddle/build
pushd Paddle/build

cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_ARM=ON -DWITH_ASCEND=OFF -DWITH_ASCEND_CL=ON -DWITH_TESTING=ON -DWITH_DISTRIBUTE=ON -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF -DPYTHON_INCLUDE_DIR=`python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"` -DWITH_CUSTOM_DEVICE=ON -DWITH_ASCEND_CXX11=ON

make TARGET=ARMV8 -j8 # or make -j8

popd
cp -R Paddle/build/paddle_inference_install_dir Paddle-Inference-Demo/c++/lib/paddle_inference
export PADDLE_INFERENCE_LIB_DIR=$(realpath Paddle-Inference-Demo/c++/lib/paddle_inference/paddle/lib)

# 编译插件
mkdir -p PaddleCustomDevice/backends/npu/build
pushd PaddleCustomDevice/backends/npu/build

# X86_64环境编译
cmake .. -DON_INFER=ON -DPADDLE_INFERENCE_LIB_DIR=${PADDLE_INFERENCE_LIB_DIR}
make -j8

# Aarch64环境编译
cmake .. -DWITH_ARM=ON -DON_INFER=ON -DPADDLE_INFERENCE_LIB_DIR=${PADDLE_INFERENCE_LIB_DIR}
make TARGET=ARMV8 -j8

# 指定插件路径
export CUSTOM_DEVICE_ROOT=$PWD
popd
```

使用 PaddleInference

```bash
pushd Paddle-Inference-Demo/c++/resnet50

# 修改 resnet50_test.cc，使用 config.EnableCustomDevice("ascend", 0) 接口替换 config.EnableUseGpu(100, 0)
  
bash run.sh
```

期待输出以下类似结果

```bash
I0516 14:40:56.197255 114531 resnet50_test.cc:74] run avg time is 115421 ms
I0516 14:40:56.197389 114531 resnet50_test.cc:89] 0 : 2.67648e-43
I0516 14:40:56.197425 114531 resnet50_test.cc:89] 100 : 1.98479e-37
I0516 14:40:56.197445 114531 resnet50_test.cc:89] 200 : 2.05547e-33
I0516 14:40:56.197463 114531 resnet50_test.cc:89] 300 : 5.06149e-42
I0516 14:40:56.197474 114531 resnet50_test.cc:89] 400 : 1.58719e-35
I0516 14:40:56.197484 114531 resnet50_test.cc:89] 500 : 7.00649e-45
I0516 14:40:56.197494 114531 resnet50_test.cc:89] 600 : 1.00972e-19
I0516 14:40:56.197504 114531 resnet50_test.cc:89] 700 : 1.92904e-23
I0516 14:40:56.197512 114531 resnet50_test.cc:89] 800 : 3.80365e-25
I0516 14:40:56.197522 114531 resnet50_test.cc:89] 900 : 1.46266e-30
```