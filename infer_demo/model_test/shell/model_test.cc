#include <gflags/gflags.h>
#include <glog/logging.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <memory>
#include <sys/time.h>
#include <paddle_inference_api.h>

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
const int CPU_THREAD_NUM = 1;

// // MODEL_NAME=mobilenet_v1_fp32_224_fluid
const std::vector<int> INPUT_SHAPE = {1, 3, 224, 224};

template <typename T>
static std::string data_to_string(const T* data, const int size) {
  std::stringstream ss;
  ss << "{";
  for (int i = 0; i < size - 1; ++i) {
    ss << data[i] << ",";
  }
  ss << data[size - 1] << "}";
  return ss.str();
}

static std::string shape_to_string(const std::vector<int>& shape) {
  std::stringstream ss;
  if (shape.empty()) {
    ss << "{}";
    return ss.str();
  }
  ss << "{";
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    ss << shape[i] << ",";
  }
  ss << shape[shape.size() - 1] << "}";
  return ss.str();
}

struct RESULT {
  int class_id;
  float score;
};

bool topk_compare_func(std::pair<float, int> a, std::pair<float, int> b) {
  return (a.first > b.first);
}

int shape_production(const std::vector<int>& shape) {
  int res = 1;
  for (auto i : shape) res *= i;
  return res;
}

double get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

std::string read_file(std::string filename) {
  std::ifstream file(filename);
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

void read_imgnp(const std::string raw_imgnp_path, float * input_data) {
  std::ifstream raw_imgnp_file(raw_imgnp_path, std::ios::in | std::ios::binary);
  if (!raw_imgnp_file) {
    std::cout << "Failed to load raw rgb image file: " <<  raw_imgnp_path << std::endl;
    return;
  }
  int64_t raw_imgnp_size = shape_production(INPUT_SHAPE);
  raw_imgnp_file.read(reinterpret_cast<char *>(input_data), raw_imgnp_size * sizeof(float));
  raw_imgnp_file.close();
}

std::vector<RESULT> postprocess(const float *output_data, int64_t output_size) {
  const int TOPK = 3;
  std::vector<std::pair<float, int>> vec;
  for (int i = 0; i < output_size; i++) {
      vec.push_back(std::make_pair(output_data[i], i));
  }
  std::partial_sort(vec.begin(), vec.begin() + TOPK, vec.end(), topk_compare_func);
  std::vector<RESULT> results(TOPK);
  for (int i = 0; i < TOPK; i++) {
      results[i].score = vec[i].first;
      results[i].class_id = vec[i].second;
  }
  return results;
}

void RunModel(const std::string model_path, const int model_type) {
  // 1. Create AnalysisConfig
  paddle_infer::Config config;
  if (model_type) {
    config.SetModel(model_path + "/__model__",
                    model_path + "/__params__");
  } else {
    config.SetModel(model_path);
  }
  // 开启 IR 优化
  config.SwitchIrOptim();
  // 启用 GPU 预测
  config.EnableUseGpu(100, 0);

  // 2. Create PaddlePredictor by AnalysisConfig
  auto predictor = paddle_infer::CreatePredictor(config);
  if (predictor == nullptr) {
    std::cout << "An internal error occurred in PaddlePredictor(AnalysisConfig)." << std::endl;
  }
  int input_num = shape_production(INPUT_SHAPE);
  std::vector<float> input_data(input_num, 1);

  // 3. Prepare input data
  auto input_names = predictor->GetInputNames();
  auto input_tensor = predictor->GetInputHandle(input_names[0]);
  input_tensor->Reshape(INPUT_SHAPE);
  input_tensor->CopyFromCpu(input_data.data());

  // 4. Warmup Run
  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }
  // 5. Repeat Run
  auto start_time = get_current_us();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor->Run();
  }
  auto end_time = get_current_us();
  // 6. Speed Report
  std::cout << "================== Speed Report ===================" << std::endl;
  std::cout << "Model: " << model_path << std::endl;
  std::cout << "Warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << (end_time - start_time) / FLAGS_repeats / 1000.0
            << " ms in average." << std::endl;

  // 5. Get all output
  auto output_names = predictor->GetOutputNames();
  int output_num = static_cast<int>(output_names.size());
  for (int i = 0; i < output_num; ++i) {
    auto output_tensor = predictor->GetInputHandle(output_names[i]);
    std::vector<int> output_shape = output_tensor->shape();
    int output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    std::cout << "Printing Output Index: <" << i << ">, shape is " << shape_to_string(output_shape) << std::endl;
    std::vector<float> output_data;
    output_data.resize(output_size);
    output_tensor->CopyToCpu(output_data.data());
    // std::cout << data_to_string<float>(output_data.data(), output_size) << std::endl;
    std::cout << "Printing Output Index: <" << i << ">, output_data size is " << output_data.size() << std::endl;
  }
  // 释放中间Tensor
  predictor->ClearIntermediateTensor();

  // 释放内存池中的所有临时 Tensor
  predictor->TryShrinkMemory();
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << "model_path\n";
    exit(1);
  }
  std::string model_path = argv[1];
  std::cout << "Model Path is <" << model_path << ">" << std::endl;
  // 0 for uncombined, 1 for combined model
  // int model_type = atoi(argv[3]);
  int model_type = 0;

  //RunModel(model_path, model_type);
  // 创建 FLOAT32 类型 DataType
  auto data_type = paddle_infer::DataType::FLOAT32;

  // 输出 data_type 的字节数 - 4
  std::cout << paddle_infer::GetNumBytesOfDataType(data_type) << std::endl;

  return 0;
}
