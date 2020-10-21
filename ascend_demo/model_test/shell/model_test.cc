#include <iostream>
#include <algorithm>
#include "paddle_api.h"
#include "logging.h"

using namespace paddle::lite_api;  // NOLINT

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
const int CPU_THREAD_NUM = 1;

struct RESULT {
  int64_t class_id;
  float score;
};

bool topk_compare_func(std::pair<float, int> a, std::pair<float, int> b) {
  return (a.first > b.first);
}

std::vector<RESULT> postprocess(const float *output_data, int64_t output_size) {
  const int64_t TOPK = std::min(static_cast<int64_t>(3), output_size);
  std::vector<std::pair<float, int>> vec;
  for (int64_t i = 0; i < output_size; ++i) {
      vec.push_back(std::make_pair(output_data[i], i));
  }
  std::partial_sort(vec.begin(), vec.begin() + TOPK, vec.end(), topk_compare_func);
  std::vector<RESULT> results(TOPK);
  for (int64_t i = 0; i < TOPK; ++i) {
      results[i].score = vec[i].first;
      results[i].class_id = vec[i].second;
  }
  return results;
}

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

void process(std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor, const std::vector<int64_t> input_shape_vec) {
  // 1. Prepare input data
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize(input_shape_vec);
  auto* input_data = input_tensor->mutable_data<float>();
  for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
    input_data[i] = 1;
  }
  // 2. Warmup Run
  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }
  // 3. Repeat Run
  auto start_time = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor->Run();
  }
  auto end_time = GetCurrentUS();
  // 4. Speed Report
  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << (end_time - start_time) / FLAGS_repeats / 1000.0
            << " ms in average.";

  // 5. Get output
  std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
  const float *output_data = output_tensor->data<float>();
  // 6. Print output
  std::vector<RESULT> results = postprocess(output_data, ShapeProduction(output_tensor->shape()));
  printf("results: %lu\n", results.size());
  for (size_t i = 0; i < results.size(); i++) {
    printf("Top%lu: %lu - %f\n", i, results[i].class_id, results[i].score);
  }
}

void RunLiteModel(std::string model_path, const std::vector<int64_t> input_shape_vec) {
  // 1. Create MobileConfig
  auto start_time = GetCurrentUS();
  MobileConfig mobile_config;
  mobile_config.set_model_from_file(model_path+".nb");
  mobile_config.set_threads(CPU_THREAD_NUM);
  mobile_config.set_power_mode(PowerMode::LITE_POWER_HIGH);
  // mobile_config.set_subgraph_model_cache_dir(model_path.substr(0, model_path.find_last_of("/")));
  mobile_config.set_device_id(1);
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor = nullptr;
  // 2. Create PaddlePredictor by MobileConfig
  try {
    predictor = CreatePaddlePredictor<MobileConfig>(mobile_config);
    std::cout << "============== RunLiteModel MobileConfig Predictor Version: " << predictor->GetVersion() << " ==============" << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(mobile config)." << std::endl;
  }
  auto end_time = GetCurrentUS();

  // 3. Run model
  process(predictor, input_shape_vec);
  LOG(INFO) << "MobileConfig preprosss: " << (end_time - start_time) / 1000.0 << " ms.";
}

#ifdef USE_FULL_API
void RunFullModel(std::string model_path, const std::vector<int64_t> input_shape_vec) {
  // 1. Create CxxConfig
  auto start_time = GetCurrentUS();
  CxxConfig cxx_config;
  cxx_config.set_model_file(model_path + "_opt/model");
  cxx_config.set_param_file(model_path + "_opt/params");
  cxx_config.set_valid_places({Place{TARGET(kX86), PRECISION(kFloat)},
                               Place{TARGET(kHost), PRECISION(kFloat)}});
  cxx_config.set_valid_places({Place{TARGET(kHuaweiAscendNPU), PRECISION(kFloat)},
                             Place{TARGET(kX86), PRECISION(kFloat)},
                             Place{TARGET(kHost), PRECISION(kFloat)}});
  // cxx_config.set_subgraph_model_cache_dir(model_path.substr(0, model_path.find_last_of("/")));
  cxx_config.set_device_id(1);
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor = nullptr;
  // 2. Create PaddlePredictor by MobileConfig
  try {
    predictor = CreatePaddlePredictor<CxxConfig>(cxx_config);
    std::cout << "============== RunFullModel CxxConfig Predictor Version: " << predictor->GetVersion() 
              << " ==============" << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }
  auto end_time = GetCurrentUS();
  // 3. Run model
  process(predictor, input_shape_vec);
  LOG(INFO) << "CXXConfig preprosss: " << (end_time - start_time) / 1000.0 << " ms.";
}

void SaveOptModel(std::string model_path, const int model_type, const std::vector<int64_t> input_shape_vec) {
  // 1. Create CxxConfig
  CxxConfig cxx_config;
  if (model_type) { // combined model
    cxx_config.set_model_file(model_path + "/model");
    cxx_config.set_param_file(model_path + "/params");
  } else {
    cxx_config.set_model_dir(model_path);
  }
  cxx_config.set_valid_places({Place{TARGET(kHuaweiAscendNPU), PRECISION(kFloat)},
                             Place{TARGET(kX86), PRECISION(kFloat)},
                             Place{TARGET(kARM), PRECISION(kFloat)},
                             Place{TARGET(kHost), PRECISION(kFloat)}});
  // cxx_config.set_subgraph_model_cache_dir(model_path.substr(0, model_path.find_last_of("/")));
  cxx_config.set_device_id(1);
  // cxx_config.set_subgraph_model_cache_dir(model_path.substr(0, model_path.find_last_of("/")));

  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<PaddlePredictor> predictor = nullptr;
  try {
    predictor = CreatePaddlePredictor<CxxConfig>(cxx_config);
    std::cout << "============== SaveOptModel CxxConfig Predictor Version: " << predictor->GetVersion() 
              << " ==============" << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }

  // 3. Save optimized model
  predictor->SaveOptimizedModel(model_path, LiteModelType::kNaiveBuffer);
  std::cout << "Save optimized model to " << (model_path+".nb") << std::endl;

  // predictor->SaveOptimizedModel(model_path+"_opt", LiteModelType::kProtobuf);
  // std::cout << "Save optimized model to " << (model_path+"_opt") << std::endl;
}
#endif

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << "model_dir model_name model_type\n";
    exit(1);
  }
  std::string model_dir = argv[1];
  std::string model_name = argv[2];
  // 0 for uncombined, 1 for combined model
  int model_type = atoi(argv[3]);

  std::vector<int64_t> input_shape_vec = {1, 3, 224, 224};

  // set input shape based on model name
  // std::vector<int64_t> input_shape_vec(4);
  // if (model_name == "mobilenet_v1") {
  //   int64_t input_shape[] = {1, 3, 224, 224};
  //   std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  // } (model_name == "mobilenet_v2") {
  //   int64_t input_shape[] = {1, 3, 224, 224};
  //   std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  // } else if (model_name == "resnet50") {
  //   int64_t input_shape[] = {1, 3, 224, 224};
  //   std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  // } else {
  //   LOG(ERROR) << "NOT supported model name!";
  //   return 0;
  // }

  LOG(INFO) << "Model Name is <" << model_name << ">, Input Shape is {" 
    << input_shape_vec[0] << ", " << input_shape_vec[1] << ", " 
    << input_shape_vec[2] << ", " << input_shape_vec[3] << "}";

  std::string model_path = model_dir + '/' + model_name;

#ifdef USE_FULL_API
  SaveOptModel(model_path, model_type, input_shape_vec);
  RunFullModel(model_path, input_shape_vec);
#endif

  RunLiteModel(model_path, input_shape_vec);

  return 0;
}
