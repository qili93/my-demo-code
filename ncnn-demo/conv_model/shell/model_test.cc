#include <iostream>
#include <sstream>
#include <algorithm>
#include <sys/time.h>
#include <vector>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include <ncnn/net.h>

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
// const int CPU_THREAD_NUM = 1;

// MODEL_NAME=squeezenet_v1.1
// const std::vector<int64_t> INPUT_SHAPE_ALIGN = {1, 3, 227, 227};

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
  std::cout << "================== Speed Report ==================" << std::endl;
  std::cout << "[ - ]  warmup: " << FLAGS_warmup 
            << ", repeats: " << FLAGS_repeats 
            << ", max=" << max << " ms, min=" << min
            << "ms, avg=" << avg << "ms" << std::endl;
}

void RunNCNNModel() {
  // 1. Load Model
  ncnn::Net mobilenet;

  mobilenet.opt.use_vulkan_compute = true;

  mobilenet.load_param("../train/torch-conv.param");
  mobilenet.load_model("../train/torch-conv.bin");

  // 2. prepare input
  ncnn::Mat input = ncnn::Mat(64, 64, 8);
  input.fill(1.f);

  std::cout << "--------- Printing Conv Input ---------" << std::endl;
  //tensor_to_string<float>(static_cast<float*>(input.data), input_shape);
  std::cout << "input.dims: " << input.dims << std::endl;
  std::cout << "input.w: " << input.w << std::endl;
  std::cout << "input.h: " << input.h << std::endl;
  std::cout << "input.c: " << input.c << std::endl;
  std::cout << "input.elemsize: " << input.elemsize << std::endl;
  std::cout << "input.elempack: " << input.elempack << std::endl;

  ncnn::Mat output;

  // 2. Warmup Run
  for (int i = 0; i < FLAGS_warmup; ++i) {
    ncnn::Extractor ex = mobilenet.create_extractor();
    ex.input("input", input);
    ex.extract("output", output);
  }

  // 3. Repeat Run
  std::vector<float> costs;
  for (int i = 0; i < FLAGS_repeats; ++i) {
    auto start_time = get_current_us();
    ncnn::Extractor ex = mobilenet.create_extractor();
    ex.input("input", input);
    ex.extract("output", output);
    auto end_time = get_current_us();
    costs.push_back((end_time - start_time) / 1000.0);
  }

  // 5. get output
  std::cout << "--------- Printing Conv Output ---------" << std::endl;
  //tensor_to_string<float>(static_cast<float*>(output.data), output_shape);
  std::cout << "output.dims: " << output.dims << std::endl;
  std::cout << "output.w: " << output.w << std::endl;
  std::cout << "output.h: " << output.h << std::endl;
  std::cout << "output.c: " << output.c << std::endl;
  std::cout << "output.elemsize: " << output.elemsize << std::endl;
  std::cout << "output.elempack: " << output.elempack << std::endl;

  // 6. print speed report
  speed_report(costs);
}

int main(int argc, char **argv) {

  RunNCNNModel();

  return 0;
}
