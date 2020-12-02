#include <iostream>
#include <sstream>
#include <algorithm>
#include <sys/time.h>
#include <vector>
#include <iomanip>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include <ncnn/net.h>

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
const int CPU_THREAD_NUM = 1;

int shape_production(const std::vector<int>& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

static inline uint64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return static_cast<uint64_t>(time.tv_sec) * 1e+6 + time.tv_usec;
}

std::string shape_to_string(const std::vector<int>& shape) {
  std::ostringstream ss;
  if (shape.empty()) {
    ss << "{}";
    return ss.str();
  }
  ss << "{";
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    ss << shape[i] << ", ";
  }
  ss << shape[shape.size() - 1] << "}";
  return ss.str();
}

template <typename T>
std::string data_to_string(const T* data, const int size) {
  std::ostringstream ss;
  ss << "[";
  for (int i = 0; i < size - 1; ++i) {
    ss << std::setprecision(3) << std::setw(10) << std::setfill(' ') 
       << std::fixed << data[i] << ", ";
  }
  ss << std::setprecision(3) << std::setw(10) << std::setfill(' ') 
     << std::fixed << data[size - 1] << "]";
  // ss << data[size - 1] << "]";
  return ss.str();
}

template <typename T>
void tensor_to_string(const T* data, const std::vector<int>& shape) {
  std::cout << "Shape: " << shape_to_string(shape) << std::endl;
  int stride = shape.back();
  int split = shape.size() > 2 ? shape[shape.size() - 2] : 0;
  int length = static_cast<int>(shape_production(shape) / stride);
  for (size_t i = 0; i < length; ++i) {
    const T * data_start = data + i * stride;
    std::cout << data_to_string<T>(data_start, stride) << std::endl;
    if (split != 0 && (i + 1) % split == 0) {
      std::cout << std::endl;
    }
  }
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
  //  set conv input - [bs, ic, ih, iw] = [1, 8, 64, 64]
  int batch_size = 1;
  int input_channel = 8;
  int input_height = 64;
  int input_width = 64;
  // set conv filter [oc, ic/groups, kh, hw] = [8, 1, 3, 3]
  int output_channel = 8;
  int groups = 8;
  int kernel_h = 3;
  int kernel_w = 3;
  // set conv attr
  int stride_h = 1;
  int stride_w = 1;
  int pad_left = 1;
  int pad_right = 1;
  int pad_top = 1;
  int pad_bottom = 1;
  int diliation = 1;
  // get shape
  const std::vector<int> input_shape = {batch_size, input_channel, input_height, input_width};
  const std::vector<int> filter_shape = {output_channel, input_channel/groups, kernel_h, kernel_w};
  const std::vector<int> bias_shape = {output_channel};
  const std::vector<int> output_shape = {batch_size, output_channel, input_height, input_width};
  // get size
  const int input_size = shape_production(input_shape);
  const int filter_size = shape_production(filter_shape);
  const int bias_size = shape_production(bias_shape);
  const int output_size = shape_production(output_shape);


  // 1. Load Model
  ncnn::Net mobilenet;
  // set opt
  mobilenet.opt.lightmode = true;
  mobilenet.opt.num_threads = CPU_THREAD_NUM;
  mobilenet.opt.use_int8_inference = false;
  mobilenet.opt.use_vulkan_compute = false;
  mobilenet.opt.use_fp16_packed = false;
  mobilenet.opt.use_fp16_storage = false;
  mobilenet.opt.use_fp16_arithmetic = false;
  mobilenet.opt.use_int8_storage = false;
  mobilenet.opt.use_int8_arithmetic = false;

  mobilenet.load_param("../train/torch-conv-64.param");
  mobilenet.load_model("../train/torch-conv-64.bin");

  // prepare input
  ncnn::Mat input = ncnn::Mat(input_height, input_width, input_channel);
  // ncnn::Mat input = ncnn::Mat(input_height, input_width, input_channel, 32UL, 8);
  for (int i = 0; i < input_size; ++i)
  {
    input[i] = 1.0f;
  }
  std::cout << "--------- Printing Conv Input ---------" << std::endl;
  // tensor_to_string<float>(static_cast<float*>(input.data), input_shape);
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
  // tensor_to_string<float>(static_cast<float*>(output.data), output_shape);
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
