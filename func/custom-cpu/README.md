# 飞桨自定义接入硬件后端(CustomCPU)

请参考以下步骤进行硬件后端(CustomCPU)的编译安装与验证。

## 一、训练示例

### 源码编译

```bash
# 下载源码，默认 develop 分支
git clone https://github.com/PaddlePaddle/PaddleCustomDevice


# 进入 backend/fakecpu 目录，并创建编译目录
cd PaddleCustomDevice/backends/fakecpu
mkdir build && cd build

# 执行CMake，如有需要可以指定Python版本
cmake -DPython_EXECUTABLE=/opt/conda/bin/python ..

# 执行编译
make -j8

# 编译步骤会自动将编译完成后的插件拷贝到插件目录，目录文件结构如下所示
/opt/conda/lib/python3.7/site-packages/paddle-plugins/
└── libpaddle-custom-cpu.so
```

### 功能验证

```bash
# 列出可用硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 预期得到如下结果
['CustomCPU']
```

## 二、推理示例

自定义硬件接入工具主要支持了 Paddle Inference 的 Python 和 C++ 两套接口。

### Python接口

编译和安装方式同第一章节的训练示例，功能验证请参考如下代码进行功能验证：

```python
TBD
```

### C++接口

#### 源码编译


#### 功能验证

## 参考文档

- [飞桨官网 - 自定义硬件接入指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/custom_device_docs/index_cn.html)