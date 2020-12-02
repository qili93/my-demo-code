#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <sys/time.h>
#include <float.h>
#include <math.h>
#include <paddle_api.h>

using namespace paddle::lite_api;  // NOLINT

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
const int CPU_THREAD_NUM = 1;
const paddle::lite_api::PowerMode CPU_POWER_MODE = paddle::lite_api::PowerMode::LITE_POWER_HIGH;

const std::string model_path = "../train/torch-conv-64/inference_model";
const std::vector<int64_t> INPUT_SHAPE = {1, 8, 64, 64};

static inline int64_t shape_production(const std::vector<int64_t>& shape) {
  int res = 1;
  for (auto i : shape) res *= i;
  return res;
}

static inline uint64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return static_cast<uint64_t>(time.tv_sec) * 1e+6 + time.tv_usec;
}

template <typename T>
std::string data_to_string(const T* data, const int64_t size) {
  std::ostringstream ss;
  ss << "[";
  for (int64_t i = 0; i < size - 1; ++i) {
    ss << std::setprecision(3) << std::setw(10) << std::setfill(' ') 
       << std::fixed << data[i] << ", ";
  }
  ss << std::setprecision(3) << std::setw(10) << std::setfill(' ') 
     << std::fixed << data[size - 1] << "]";
  // ss << data[size - 1] << "]";
  return ss.str();
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
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
void tensor_to_string(const T* data, const std::vector<int64_t>& shape) {
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

void process(std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor) {
  // 1. Prepare input data
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize(INPUT_SHAPE);
  auto* input_data = input_tensor->mutable_data<float>();
  for (int64_t i = 0; i < shape_production(INPUT_SHAPE); ++i) {
    input_data[i] = i + 1.0f;
  }

  // 2. Warmup Run
  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }
  // 3. Repeat Run
  std::vector<float> costs;
  for (int i = 0; i < FLAGS_repeats; ++i) {
    auto start_time = get_current_us();
    predictor->Run();
    auto end_time = get_current_us();
    costs.push_back((end_time - start_time) / 1000.0);
  }

  // 4. Get all output
  std::cout << std::endl << "Input Index: <0>" << std::endl;
  // tensor_to_string<float>(input_data, input_tensor->shape());
  int output_num = static_cast<int>(predictor->GetOutputNames().size());
  for (int i = 0; i < output_num; ++i) {
    std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(i)));
    const float *output_data = output_tensor->data<float>();
    std::cout << "Output Index: <" << i << ">" << std::endl;
    // tensor_to_string<float>(output_data, output_tensor->shape());
  }

  // 5. speed report
  speed_report(costs);
}

void RunLiteModel(const std::string model_path) {
  // 1. Create MobileConfig
  auto start_time = get_current_us();
  MobileConfig mobile_config;
  mobile_config.set_model_from_file(model_path+".nb");
  // Load model from buffer
  // std::string model_buffer = ReadFile(model_path+".nb");
  // mobile_config.set_model_from_buffer(model_buffer);
  mobile_config.set_threads(CPU_THREAD_NUM);
  mobile_config.set_power_mode(PowerMode::LITE_POWER_HIGH);
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor = nullptr;
  // 2. Create PaddlePredictor by MobileConfig
  try {
    predictor = CreatePaddlePredictor<MobileConfig>(mobile_config);
    std::cout << "MobileConfig Predictor Version: " << predictor->GetVersion() << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(mobile config)." << std::endl;
  }
  auto end_time = get_current_us();

  // 3. Run model
  process(predictor);
  std::cout << "MobileConfig preprosss: " << (end_time - start_time) / 1000.0 << " ms." << std::endl;
}

#ifdef USE_FULL_API
void RunFullModel(const std::string model_path) {
  // 1. Create CxxConfig
  auto start_time = get_current_us();
  CxxConfig cxx_config;
  cxx_config.set_model_file(model_path + "_opt/model");
  cxx_config.set_param_file(model_path + "_opt/params");
  cxx_config.set_valid_places({Place{TARGET(kX86), PRECISION(kFloat)},
                               Place{TARGET(kHost), PRECISION(kFloat)}});
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor = nullptr;
  // 2. Create PaddlePredictor by MobileConfig
  try {
    predictor = CreatePaddlePredictor<CxxConfig>(cxx_config);
    std::cout << "CxxConfig Predictor Version: " << predictor->GetVersion() << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }
  auto end_time = get_current_us();
  // 3. Run model
  process(predictor);
  std::cout << "CxxConfig preprosss: " << (end_time - start_time) / 1000.0 << " ms." << std::endl;
}

void SaveOptModel(const std::string model_path, const int model_type = 0) {
  // 1. Create CxxConfig
  CxxConfig cxx_config;
  if (model_type) { // combined model
    cxx_config.set_model_file(model_path + "/__model__");
    cxx_config.set_param_file(model_path + "/__params__");
  } else {
    cxx_config.set_model_dir(model_path);
  }
  cxx_config.set_valid_places({Place{TARGET(kX86), PRECISION(kFloat)},
                               Place{TARGET(kHost), PRECISION(kFloat)}});
  // cxx_config.set_subgraph_model_cache_dir(model_path.substr(0, model_path.find_last_of("/")));

  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<PaddlePredictor> predictor = nullptr;
  try {
    predictor = CreatePaddlePredictor<CxxConfig>(cxx_config);
    std::cout << "CxxConfig Predictor Version: " << predictor->GetVersion() << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }

  // 3. Save optimized model
  predictor->SaveOptimizedModel(model_path, LiteModelType::kNaiveBuffer);
  std::cout << "Save optimized model to " << (model_path+".nb") << std::endl;

  predictor->SaveOptimizedModel(model_path+"_opt", LiteModelType::kProtobuf);
  std::cout << "Save optimized model to " << (model_path+"_opt") << std::endl;
}
#endif

int main(int argc, char **argv) {
  // if (argc < 2) {
  //   std::cerr << "[ERROR] usage: ./" << argv[0] << "model_path\n";
  //   exit(1);
  // }
  // const std::string model_path = argv[1];
  // std::cout << "Model Path is <" << model_path << ">" << std::endl;

  // // 0 for umcombined
  // const int model_type = 0;

#ifdef USE_FULL_API
  SaveOptModel(model_path);
  RunFullModel(model_path);
#endif

  RunLiteModel(model_path);

  return 0;
}
