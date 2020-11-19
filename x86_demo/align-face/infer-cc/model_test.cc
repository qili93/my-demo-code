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

// // MODEL_NAME=align150-fp32
const std::vector<int> INPUT_SHAPE = {1, 3, 128, 128};

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

int ShapeProduction(const std::vector<int>& shape) {
  int res = 1;
  for (auto i : shape) res *= i;
  return res;
}

double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

void read_imgnp(const std::string raw_imgnp_path, float * input_data) {
  std::ifstream raw_imgnp_file(raw_imgnp_path, std::ios::in | std::ios::binary);
  if (!raw_imgnp_file) {
    std::cout << "Failed to load raw rgb image file: " <<  raw_imgnp_path << std::endl;
    return;
  }
  int64_t raw_imgnp_size = ShapeProduction(INPUT_SHAPE);
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

void RunModel(const std::string model_path, const std::string image_path, const int model_type) {
  // 1. Create AnalysisConfig
  paddle::AnalysisConfig config;
  if (model_type) {
    config.SetModel(model_path + "/__model__",
                    model_path + "/__params__");
  } else {
    config.SetModel(model_path);
  }
  // use ZeroCopyTensor, Must be set to false
  config.SwitchUseFeedFetchOps(false);
  config.SetCpuMathLibraryNumThreads(CPU_THREAD_NUM);
  // turn off for int8 model
  // config.SwitchIrOptim(false);

  // 2. Create PaddlePredictor by AnalysisConfig
  auto predictor = paddle::CreatePaddlePredictor(config);
  if (predictor == nullptr) {
    std::cout << "An internal error occurred in PaddlePredictor(AnalysisConfig)." << std::endl;
  }
  int nums = ShapeProduction(INPUT_SHAPE);
  float* input = new float[nums];
  // for (int i = 0; i < nums; ++i) input[i] = 1.f;
  read_imgnp(image_path, input);

  // 3. Prepare input data
  auto input_names = predictor->GetInputNames();
  auto input_tensor = predictor->GetInputTensor(input_names[0]);
  input_tensor->Reshape(INPUT_SHAPE);
  input_tensor->copy_from_cpu(input);

  // 4. Warmup Run
  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->ZeroCopyRun();
  }
  // 5. Repeat Run
  auto start_time = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor->ZeroCopyRun();
  }
  auto end_time = GetCurrentUS();
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
    auto output_tensor = predictor->GetOutputTensor(output_names[i]);
    std::vector<int> output_shape = output_tensor->shape();
    int output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    std::cout << "Printing Output Index: <" << i << ">, shape is " << shape_to_string(output_shape) << std::endl;
    std::vector<float> output_data;
    output_data.resize(output_size);
    output_tensor->copy_to_cpu(output_data.data());
    std::cout << data_to_string<float>(output_data.data(), output_size) << std::endl;
  }

  // 9. free memory
  delete[] input;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << "assets_dir model_name image_name\n";
    exit(1);
  }
  std::string assets_dir = argv[1];
  std::string model_name = argv[2];
  std::string image_name = argv[3];
  // 0 for uncombined, 1 for combined model
  // int model_type = atoi(argv[3]);
  int model_type = 1;

  std::string model_path = assets_dir + "/models/" + model_name;
  std::string image_path = assets_dir + "/images/" + image_name;

  std::cout << "Model Path is <" << model_path << ">" << std::endl;
  std::cout << "Image Path is <" << image_path << ">" << std::endl;

  RunModel(model_path, image_path, model_type);
  return 0;
}
