# README

## 1. Run PaddlePaddle-IPU docker container

**Step 1: Pull PaddlePaddle IPU docker image**

```bash
# Pull PaddlePaddle IPU docker image
docker pull registry.baidubce.com/device/paddlepaddle:ipu-poplar250
```

**Step 2. Start a new container based on the new image**

```bash
# Note: replace /home/liqi27 to your home directory
export IPUOF_CONFIG_PATH=/opt/ipuof.conf
docker run -it --name paddle-ipu -v /home/liqi27:/workspace \
     --shm-size=128G --network=host --ulimit memlock=-1:-1 \
     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
     --cap-add=IPC_LOCK --device=/dev/infiniband/ --ipc=host \
     -v ${IPUOF_CONFIG_PATH}:/ipuof.conf -e IPUOF_CONFIG_PATH=/ipuof.conf \
     registry.baidubce.com/device/paddlepaddle:ipu-poplar250 /bin/bash
```

**Step 3. Verify paddlepaddle-ipu program**

```bash
# Verify IPU can be monitor inside container
gc-info -l && gc-monitor
# Expected to get output as following
Graphcore device listing:
Partition: ipuof [active]
-+- Id: [0], target: [Fabric], IPU-M host: [172.31.100.130], IPU#: [3]
-+- Id: [1], target: [Fabric], IPU-M host: [172.31.100.130], IPU#: [2]
-+- Id: [2], target: [Fabric], IPU-M host: [172.31.100.130], IPU#: [1]
-+- Id: [3], target: [Fabric], IPU-M host: [172.31.100.130], IPU#: [0]
-+- Id: [4], target: [Multi IPU]
 |--- Id: [0], DNC Id: [0], IPU-M host: [172.31.100.130], IPU#: [3]
 |--- Id: [1], DNC Id: [1], IPU-M host: [172.31.100.130], IPU#: [2]
-+- Id: [5], target: [Multi IPU]
 |--- Id: [2], DNC Id: [0], IPU-M host: [172.31.100.130], IPU#: [1]
 |--- Id: [3], DNC Id: [1], IPU-M host: [172.31.100.130], IPU#: [0]
-+- Id: [6], target: [Multi IPU]
 |--- Id: [0], DNC Id: [0], IPU-M host: [172.31.100.130], IPU#: [3]
 |--- Id: [1], DNC Id: [1], IPU-M host: [172.31.100.130], IPU#: [2]
 |--- Id: [2], DNC Id: [2], IPU-M host: [172.31.100.130], IPU#: [1]
 |--- Id: [3], DNC Id: [3], IPU-M host: [172.31.100.130], IPU#: [0]

+---------------+--------------------------------------------------------------------------------+
|  gc-monitor   |              Partition: ipuof [active] has 4 reconfigurable IPUs               |
+-------------+--------------------+--------+--------------+----------+------+----+------+-------+
|    IPU-M    |       Serial       |IPU-M SW|Server version|  ICU FW  | Type | ID | IPU# |Routing|
+-------------+--------------------+--------+--------------+----------+------+----+------+-------+
|...31.100.130| 0134.0002.8210321  |        |    1.8.1     |  2.3.5   |M2000 | 0  |  3   |  DNC  |
|...31.100.130| 0134.0002.8210321  |        |    1.8.1     |  2.3.5   |M2000 | 1  |  2   |  DNC  |
|...31.100.130| 0134.0001.8210321  |        |    1.8.1     |  2.3.5   |M2000 | 2  |  1   |  DNC  |
|...31.100.130| 0134.0001.8210321  |        |    1.8.1     |  2.3.5   |M2000 | 3  |  0   |  DNC  |
+-------------+--------------------+--------+--------------+----------+------+----+------+-------+
+--------------------------------------------------------------------------------------------------+
|                             No attached processes in partition ipuof                             |
+--------------------------------------------------------------------------------------------------+

# Verify paddlepaddle-ipu is installed
pip list | grep paddlepaddle-ipu
# Expected to get output as following
paddlepaddle-ipu       0.0.0.dev250

# Verify paddlepaddle-ipu works successfully
python -c "import paddle; paddle.utils.run_check()"
# Expected to get output as following
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 CPU.
PaddlePaddle works well on 2 CPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## 2. Training and inference sample on IPU

**Step 1: Download sample code and unzip**

```bash
# Download and unzip sample code for paddlepaddle-ipu
wget https://paddle-device.bj.bcebos.com/ipu/sample.tar.gz
tar -zxvf sample.tar.gz && cd sample
# list contents of sample as following after unzip
sample/
├── image
│   └── infer_3.png
├── mnist_infer.py
├── mnist_train.py
├── model
└── README.md
```

**Step 2: Run MNIST train**

```bash
# Run minst train
python mnist_train.py
# Expected to get output as following
start training
start compiling model for ipu, it will need some minutes
Graph compilation: 20/100
... ...
Graph compilation: 100/100
finish model compiling!
step: 0, loss: 2.798431158065796
step: 40, loss: 0.684889018535614
... ...
step: 920, loss: 0.3129602372646332
finish training!
start verifying
start compiling model for ipu, it will need some minutes
finish model compiling!
top1 score: 0.8819110576923077
# Also inference model will be saved under ./model directory
# list contents of sample as following after model train
sample/
├── image
│   └── infer_3.png
├── mnist_infer.py
├── mnist_train.py
├── model
│   ├── mnist.pdiparams
│   └── mnist.pdmodel
└── README.md
```

**Step 3: Run MNIST inference**

```bash
# Run minst train
python mnist_train.py
# Expected to get output as following
--- Running analysis [ir_graph_build_pass]
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [inference_process_pass]
I0515 16:20:35.537405 25728 ir_analysis_pass.cc:46] argument has no fuse statis
--- Running analysis [ir_params_sync_among_devices_pass]
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]
I0515 16:20:35.538944 25728 analysis_predictor.cc:1024] ======= optimize end =======
I0515 16:20:35.539006 25728 naive_executor.cc:102] ---  skip [feed], feed -> img
I0515 16:20:35.539033 25728 naive_executor.cc:102] ---  skip [save_infer_model/scale_0.tmp_0], fetch -> fetch
Inference result of ./infer_3.png is:  3
```


