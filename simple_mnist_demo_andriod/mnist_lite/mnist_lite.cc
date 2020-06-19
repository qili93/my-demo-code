#include <iostream>
#include <algorithm>
#include "paddle_api.h"

using namespace paddle::lite_api;  // NOLINT

const int CPU_THREAD_NUM = 1;

struct RESULT {
  int class_id;
  float score;
};

bool topk_compare_func(std::pair<float, int> a, std::pair<float, int> b) {
  return (a.first > b.first);
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

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

void RunModel(std::string model_name) {
  // 1. Create MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_name);
  config.set_threads(CPU_THREAD_NUM);
  config.set_power_mode(PowerMode::LITE_POWER_HIGH);
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);
  std::cout << "PaddlePredictor Version: " << predictor->GetVersion() << std::endl;
  // 3. Prepare input data
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, 1, 28, 28});
  auto* input_data = input_tensor->mutable_data<float>();
  for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
    input_data[i] = 1;
  }
  // 4. Run predictor
  predictor->Run();
  // 5. Get output
  std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
  const float *output_data = output_tensor->data<float>();
  std::vector<RESULT> results = postprocess(output_data, ShapeProduction(output_tensor->shape()));
  // 6. Print output
  printf("results: %lu\n", results.size());
  for (int i = 0; i < results.size(); i++) {
    printf("Top%d: %d - %f\n", i, results[i].class_id, results[i].score);
  }
}

void SaveModel(std::string model_dir) {
    // 1. Create CxxConfig
    CxxConfig config;
    config.set_model_dir(model_dir);
    config.set_threads(CPU_THREAD_NUM);
    config.set_power_mode(PowerMode::LITE_POWER_HIGH);
    config.set_valid_places({Place{TARGET(kNPU), PRECISION(kFloat)},
                             Place{TARGET(kARM), PRECISION(kFloat)}});
    config.set_subgraph_model_cache_dir(model_dir.substr(0, model_dir.find_last_of("/")));
    // 2. Create PaddlePredictor by CxxConfig
    std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<CxxConfig>(config);
    std::cout << "PaddlePredictor Version: " << predictor->GetVersion() << std::endl;
    // 3. Save optimized model
    predictor->SaveOptimizedModel(model_dir, LiteModelType::kNaiveBuffer);
    std::cout << "Load model from " << model_dir << std::endl;
    std::cout << "Save optimized model to " << (model_dir+".nb") << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << " model_dir [save|predict]\n";
    exit(1);
  }
  std::string model_path = argv[1];
  std::string exe_mode = argv[2];
  if (exe_mode == "save") {
#ifdef USE_FULL_API
      SaveModel(model_path);
#else
    std::cerr << "[ERROR] model " << argv[2] << " is NOT supported on tiny publish\n";
    exit(1);
#endif
  }
  if (exe_mode == "predict") {
      RunModel(model_path);
  }
  return 0;
}

