#include <iostream>
#include <algorithm>
#include <thread>
#include "paddle_api.h"

using namespace paddle::lite_api;  // NOLINT

const int CPU_THREAD_NUM = 1;
const std::vector<int64_t> INPUT_SHAPE = {1, 1, 28, 28};

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

void process(std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor) {
    // 1. Prepare input data
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize(INPUT_SHAPE);
  auto* input_data = input_tensor->mutable_data<float>();
  for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
    input_data[i] = 1;
  }
  // 2. Run predictor
  predictor->Run();
  // 3. Get output
  std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
  const float *output_data = output_tensor->data<float>();
  std::vector<RESULT> results = postprocess(output_data, ShapeProduction(output_tensor->shape()));
  // 4. Print output
  printf("results: %du\n", results.size());
  for (int i = 0; i < results.size(); i++) {
    printf("Top%d: %d - %f\n", i, results[i].class_id, results[i].score);
  }
}

void RunModel(const std::string& thread_name, const std::string model_dir, const int device_id) {
  // 1. Create CxxConfig
  MobileConfig mobile_config;
  mobile_config.set_model_from_file(model_dir+".nb");
  mobile_config.set_threads(CPU_THREAD_NUM);
  mobile_config.set_power_mode(PowerMode::LITE_POWER_HIGH);
  //cxx_config.set_subgraph_model_cache_dir(model_dir.substr(0, model_dir.find_last_of("/")));
  mobile_config.set_device_id(device_id);
  //cxx_config.set_subgraph_model_cache_dir("/data/local/tmp");
  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  try {
    predictor = CreatePaddlePredictor<MobileConfig>(mobile_config);
    std::cout << "[" << thread_name << "]: PaddlePredictor Version: " << predictor->GetVersion() << std::endl;
  } catch (std::exception e) {
    std::cout << "[" << thread_name << "]: An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }
  // 3. Run model
  process(predictor);
}

void SaveRunModel(const std::string& thread_name, const std::string model_dir, const int device_id) {
  // 1. Create CxxConfig
  CxxConfig cxx_config;
  cxx_config.set_model_dir(model_dir);
  cxx_config.set_threads(CPU_THREAD_NUM);
  cxx_config.set_power_mode(PowerMode::LITE_POWER_HIGH);
  cxx_config.set_valid_places({Place{TARGET(kHuaweiAscendNPU), PRECISION(kFloat)},
                             Place{TARGET(kX86), PRECISION(kFloat)},
                             Place{TARGET(kHost), PRECISION(kFloat)}});
  //cxx_config.set_subgraph_model_cache_dir(model_dir.substr(0, model_dir.find_last_of("/")));
  cxx_config.set_device_id(device_id);
  //cxx_config.set_subgraph_model_cache_dir("/data/local/tmp");
  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  try {
    predictor = CreatePaddlePredictor<CxxConfig>(cxx_config);
    std::cout << "[" << thread_name << "]: PaddlePredictor Version: " << predictor->GetVersion() << std::endl;
  } catch (std::exception e) {
    std::cout << "[" << thread_name << "]: An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }
  // 3. Run model
  process(predictor);
  // 4. Save optimized model
  predictor->SaveOptimizedModel(model_dir, LiteModelType::kNaiveBuffer);
  //predictor->SaveOptimizedModel(model_dir, LiteModelType::kProtobuf);
  std::cout << "[" << thread_name << "]: Load model from " << model_dir << std::endl;
  std::cout << "[" << thread_name << "]: Save optimized model to " << (model_dir+".nb") << std::endl;
}

int main() {
    std::thread t1(SaveRunModel, "t1", "../assets/models/mnist_model1", 1);
    std::thread t2(RunModel, "t2", "../assets/models/mnist_model2", 2);
    t1.join();
    t2.join();

    std::this_thread::sleep_for(std::chrono::milliseconds{5000});
    return 0;
}

// int main(int argc, char **argv) {
//   std::thread thread1(RunModel, "../assets/models/mnist_model1", 1);
//   std::thread thread2(RunModel, "../assets/models/mnist_model2", 2);
//   thread1.join();
//   thread2.join();

//   std::this_thread::sleep_for(std::chrono::milliseconds{5000});
//   return 0;
// }

