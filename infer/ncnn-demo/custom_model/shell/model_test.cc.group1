#include <iostream>
#include <sstream>
#include <algorithm>
#include <sys/time.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include <ncnn/net.h>

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
const int CPU_THREAD_NUM = 1;

static inline int shape_production(const std::vector<int>& shape) {
  int res = 1;
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
std::string data_to_string(const T* data, const int64_t size) {
  std::ostringstream ss;
  ss << "[";
  for (int64_t i = 0; i < size - 1; ++i) {
    ss << std::setprecision(2) << std::setw(9) << std::setfill(' ') 
       << std::fixed << data[i] << ", ";
  }
  ss << std::setprecision(2) << std::setw(9) << std::setfill(' ') 
     << std::fixed << data[size - 1] << "]";
  // ss << data[size - 1] << "]";
  return ss.str();
}

template <typename T>
void tensor_to_string(const T* data, const std::vector<int>& shape) {
  std::cout << "Shape: " << shape_to_string(shape) << std::endl;
  int64_t stride = shape.back();
  int64_t split = shape.size() > 2 ? shape[shape.size() - 2] : 0;
  int64_t length = static_cast<int64_t>(shape_production(shape) / stride);
  for (size_t i = 0; i < length; ++i) {
    const T * data_start = data + i * stride;
    std::cout << data_to_string<T>(data_start, stride) << std::endl;
    if (split != 0 && i % split == 1) {
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

void RunNCNNOP() {
    //  set conv input - [bs, ic, ih, iw] = [1, 3, 2, 2]
    int batch_size = 1;
    int input_channel = 3;
    int input_height = 2;
    int input_width = 2;
    // set conv filter [oc, ic/groups, kh, hw] = [2, 3, 1, 1]
    int output_channel = 2;
    int groups = 1;
    int kernel_h = 1;
    int kernel_w = 1;
    // set conv attr
    int stride_h = 1;
    int stride_w = 1;
    int padding = 0;
    int diliation = 1;

    // get shape
    const std::vector<int> input_shape = {batch_size, input_channel, input_height, input_width};
    const std::vector<int> filter_shape = {output_channel, input_channel/groups, kernel_w, kernel_h};
    const std::vector<int> bias_shape = {output_channel};
    const std::vector<int> output_shape = {batch_size, output_channel, 2, 2};
    // get size
    const int input_size = shape_production(input_shape);
    const int filter_size = shape_production(filter_shape);
    const int bias_size = shape_production(bias_shape);
    const int output_size = shape_production(output_shape);

    ncnn::Option opt;
    opt.num_threads = 1;

    ncnn::Layer* op = ncnn::create_layer("ConvolutionDepthWise");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, output_channel); // output_channel = 2
    pd.set(1, kernel_w); // kernel_w = 1
    pd.set(11, kernel_h); // kernel_h = 1
    pd.set(2, diliation); // dilation_w = 1
    pd.set(12, diliation); // dilation_h = 1
    pd.set(3, stride_w); // stride_w = 1
    pd.set(13, stride_h); // stride_h = 1
    pd.get(4, padding); // pad_w = 0
    pd.get(14, padding); // pad_h = 0
    pd.set(5, 1);// bias_term  = 0
    pd.set(6, filter_size);// weight_data_size = [2, 3, 1, 1] = [oc, ic/groups, kw, kw]
    pd.set(7, groups);// group = 1

    op->load_param(pd);

    ncnn::Mat weights[2];
    // filter
    weights[0].create(filter_size);
    for (int i = 0; i < filter_size; ++i)
    {
        weights[0][i] = std::pow(10, i)/100.f;
    }
    std::cout << "--------- Printing Conv Filter ---------" << std::endl;
    tensor_to_string<float>(static_cast<float*>(weights[0].data), filter_shape);
    // bias
    weights[1].create(bias_size);
    for (int i = 0; i < bias_size; ++i)
    {
        weights[1][i] = i + 1.f;
    }
    std::cout << "--------- Printing Conv Bias ---------" << std::endl;
    tensor_to_string<float>(static_cast<float*>(weights[1].data), bias_shape);

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    // prepare input
    ncnn::Mat input = ncnn::Mat(input_height, input_width, input_channel);
    for (int i = 0; i < input_size; ++i)
    {
      input[i] = i + 1.f;
    }
    std::cout << "--------- Printing Conv Input ---------" << std::endl;
    tensor_to_string<float>(static_cast<float*>(input.data), input_shape);

    ncnn::Mat output;

    // foward - warmup
    for (int i = 0; i < FLAGS_warmup; ++i) {
      op->forward(input, output, opt);
    }
    // foward - repeat
    std::vector<float> costs;
    for (int i = 0; i < FLAGS_repeats; ++i) {
      auto start_time = get_current_us();
      op->forward(input, output, opt);
      auto end_time = get_current_us();
      costs.push_back((end_time - start_time) / 1000.0);
    }
    
    // get output
    std::cout << "--------- Printing Conv Output ---------" << std::endl;
    tensor_to_string<float>(static_cast<float*>(output.data), output_shape);

    op->destroy_pipeline(opt);

    delete op;

    // print speed report
    speed_report(costs);
}

int main(int argc, char **argv) {

  RunNCNNOP();

  return 0;
}
