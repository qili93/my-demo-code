#include <iostream>
#include <sstream>
#include <algorithm>
#include <sys/time.h>
#include <vector>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include <MNN/MNNDefine.h>
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

const int FLAGS_warmup = 0;
const int FLAGS_repeats = 1;
const int CPU_THREAD_NUM = 1;

// MODEL_NAME=squeezenet_v1.1
// const std::vector<int64_t> INPUT_SHAPE_ALIGN = {1, 3, 227, 227};

const std::string modelPath = "../models/MobileNetV2.mnn";

int64_t shape_production(const std::vector<int64_t>& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

static inline uint64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return static_cast<uint64_t>(time.tv_sec) * 1e+6 + time.tv_usec;
}

void speed_report(const std::vector<float>& costs) {
  float max = 0, min = FLT_MAX, sum = 0, avg;
  for (auto v : costs) {
      max = fmax(max, v);
      min = fmin(min, v);
      sum += v;
  }
  avg = costs.size() > 0 ? sum / costs.size() : 0;
  std::cout << "================== Speed Report ===================" << std::endl;
  std::cout << "Warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats 
            << ", max = " << max << " ms, min = " << min
            << "ms , avg = " << avg << "ms" << std::endl;
  //printf("[ - ] %-24s    max = %8.3fms  min = %8.3fms  avg = %8.3fms\n", name.c_str(), max, avg == 0 ? 0 : min, avg);
}

void RunMNNModel() {
  // 1. Load Model
  std::shared_ptr<MNN::Interpreter> net = 
      std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelPath.c_str()));
  // 2. create session
  MNN::ScheduleConfig config;
  config.numThread = CPU_THREAD_NUM;
  config.type = MNN_FORWARD_CPU;
  MNN::BackendConfig backendConfig;
  backendConfig.precision = MNN::BackendConfig::Precision_High;
  backendConfig.power = MNN::BackendConfig::Power_High;
  config.backendConfig = &backendConfig;
  auto session         = net->createSession(config);

  // 3. reset all inputs
  auto allInput = net->getSessionInputAll(session);
  for (auto& iter : allInput) {
      auto inputTensor = iter.second;
      auto size = inputTensor->size();
      if (size <= 0) {
          continue;
      }
      MNN::Tensor tempTensor(inputTensor, inputTensor->getDimensionType());
      ::memset(tempTensor.host<void>(), 0, tempTensor.size());
      inputTensor->copyFromHostTensor(&tempTensor);
  }

  // 4. Write input tensor
  auto inputTensor = net->getSessionInput(session, NULL);
  std::shared_ptr<MNN::Tensor> givenTensor(MNN::Tensor::createHostTensorFromDevice(inputTensor, false));
  // set data to given tensor
  auto givenData   = givenTensor->host<float>();
  for (int i = 0; i < givenTensor->elementSize(); ++i) {
    givenData[i] = 1.0f;
  }
  // output tensor
  auto outputTensor = net->getSessionOutput(session, NULL);
  std::shared_ptr<MNN::Tensor> expectTensor(MNN::Tensor::createHostTensorFromDevice(outputTensor, false));

  // 5. Warmup Run
  for (int i = 0; i < FLAGS_warmup; ++i) {
    inputTensor->copyFromHostTensor(givenTensor.get());
    net->runSession(session);
    outputTensor->copyToHostTensor(expectTensor.get());
  }

  // 6. Repeat Run
  std::vector<float> costs;
  for (int i = 0; i < FLAGS_repeats; ++i) {
    auto start_time = get_current_us();
    inputTensor->copyFromHostTensor(givenTensor.get());
    net->runSession(session);
    outputTensor->copyToHostTensor(expectTensor.get());
    auto end_time = get_current_us();
    costs.push_back((end_time - start_time) / 1000.0);
  }
  speed_report(costs);
  
  // 7. get output
  auto outputData = expectTensor->host<float>();
  for (int i = 0; i < 10; ++i) {
    std::cout << "Output <" << i << "> = " << outputData[i] << std::endl;
  }
}

int main(int argc, char **argv) {

  RunMNNModel();

  return 0;
}
