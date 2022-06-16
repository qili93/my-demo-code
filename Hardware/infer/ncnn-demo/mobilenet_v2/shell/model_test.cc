#include <iostream>
#include <sstream>
#include <algorithm>
#include <sys/time.h>
#include <vector>
#include <stdio.h>

#include <ncnn/net.h>

const int FLAGS_warmup = 0;
const int FLAGS_repeats = 1;
// const int CPU_THREAD_NUM = 1;

// MODEL_NAME=squeezenet_v1.1
// const std::vector<int64_t> INPUT_SHAPE_ALIGN = {1, 3, 227, 227};

int64_t shape_production(const std::vector<int64_t>& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

double get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

void RunNCNNModel() {
  // 1. Load Model
  ncnn::Net mobilenet;

  mobilenet.opt.use_vulkan_compute = true;

  mobilenet.load_param("../models/MobileNetV2-opt.param");
  mobilenet.load_model("../models/MobileNetV2-opt.bin");

  // 2. prepare input
  ncnn::Mat in = ncnn::Mat(224, 224, 3);
  in.fill(1.f);

  ncnn::Mat out;

  // 2. Warmup Run
  for (int i = 0; i < FLAGS_warmup; ++i) {
    ncnn::Extractor ex = mobilenet.create_extractor();
    ex.input("0", in);
    ex.extract("443", out);
  }

  // 3. Repeat Run
  auto start_time = get_current_us();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "=====================" << std::endl;
    ncnn::Extractor ex = mobilenet.create_extractor();
    ex.input("0", in);
    ex.extract("443", out);
  }
  auto end_time = get_current_us();
  // 4. Speed Report
  std::cout << "================== Speed Report ===================" << std::endl;
  std::cout << "Warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << (end_time - start_time) / FLAGS_repeats / 1000.0
            << " ms in average." << std::endl;

  // 5. get output
  std::vector<float> cls_scores;
  cls_scores.resize(out.w);
  for (int j = 0; j < out.w; j++)
  {
    cls_scores[j] = out[j];
  }
}

int main(int argc, char **argv) {

  RunNCNNModel();

  return 0;
}
