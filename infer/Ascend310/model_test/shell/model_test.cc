#include <iostream>
#include <algorithm>
#include <sys/time.h>
#include <paddle_api.h>

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
const int CPU_THREAD_NUM = 1;

// model - inception_v4
const std::vector<int64_t> INPUT_SHAPE = {1, 3, 224, 224};

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

int64_t ShapeProduction(const std::vector<int64_t>& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

void process(std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor) {
  // 1. Prepare input data
  std::unique_ptr<paddle::lite_api::Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize(INPUT_SHAPE);
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
  std::cout << "================== Speed Report ===================" << std::endl;
  std::cout << "Warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats 
            << ", spend " << (end_time - start_time) / FLAGS_repeats / 1000.0
            << " ms in average." << std::endl;

  // 5. Get output (1 x 1000)
  std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(std::move(predictor->GetOutput(0)));
  const float *output_data = output_tensor->data<float>();

  // 6. Print TOPK
  std::vector<RESULT> results = postprocess(output_data, ShapeProduction(output_tensor->shape()));
  printf("results: %lu\n", results.size());
  for (size_t i = 0; i < results.size(); i++) {
    std::cout << "Top" << i << ": " << results[i].class_id << " - " << results[i].score << std::endl;
    // printf("Top%lu: %lu - %f\n", i, results[i].class_id, results[i].score);
  }
}

void RunLiteModel(std::string model_path) {
  // 1. Create MobileConfig
  auto start_time = GetCurrentUS();
  paddle::lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(model_path+".nb");
  mobile_config.set_threads(CPU_THREAD_NUM);
  mobile_config.set_power_mode(paddle::lite_api::PowerMode::LITE_POWER_HIGH);
  // mobile_config.set_subgraph_model_cache_dir(model_path.substr(0, model_path.find_last_of("/")));
  mobile_config.set_device_id(1);
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  try {
    predictor = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(mobile_config);
    std::cout << "Predictor Version: " << predictor->GetVersion() << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(mobile config)." << std::endl;
  }
  auto end_time = GetCurrentUS();
  std::cout << "MobileConfig preprosss: " << (end_time - start_time) / 1000.0 << " ms." << std::endl;

  // 3. Run model
  process(predictor);
}

#ifdef USE_FULL_API
void RunFullModel(std::string model_path) {
  // 1. Create CxxConfig
  auto start_time = GetCurrentUS();
  paddle::lite_api::CxxConfig cxx_config;
  cxx_config.set_model_dir(model_path);
  cxx_config.set_valid_places({paddle::lite_api::Place{TARGET(kHuaweiAscendNPU), PRECISION(kFloat)},
                               paddle::lite_api::Place{TARGET(kX86), PRECISION(kFloat)}});
  // cxx_config.set_subgraph_model_cache_dir(model_path.substr(0, model_path.find_last_of("/")));
  cxx_config.set_device_id(1);

  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  try {
    predictor = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::CxxConfig>(cxx_config);
    std::cout << "Predictor Version: " << predictor->GetVersion() << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }
  auto end_time = GetCurrentUS();
  std::cout << "CxxConfig preprosss: " << (end_time - start_time) / 1000.0 << " ms." << std::endl;

  // 3. Run model
  process(predictor);

  // 4. Save optimized model
  predictor->SaveOptimizedModel(model_path, paddle::lite_api::LiteModelType::kNaiveBuffer);
  std::cout << "Save optimized model to " << (model_path+".nb") << std::endl;
}
#endif

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << "model_path\n";
    exit(1);
  }
  std::string model_path = argv[1];
  std::cout << "Model Path is <" << model_path << ">" << std::endl;

#ifdef USE_FULL_API
  RunFullModel(model_path);
#endif

  RunLiteModel(model_path);

  return 0;
}
