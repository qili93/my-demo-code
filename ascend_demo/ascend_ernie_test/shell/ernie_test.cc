#include <iostream>
#include <algorithm>
#include <sys/time.h>
#include "paddle_api.h"
#include "logging.h"

using namespace paddle::lite_api;  // NOLINT

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;

const int CPU_THREAD_NUM = 1;
const std::vector<int64_t> INPUT_SHAPE = {1, 128, 1};

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

double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

bool hasEnding(std::string const &fullString, std::string const &ending) {
    if (fullString.length() < ending.length()) return false;
    return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
}

void process(std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor) {
  // 1. Prepare input data
  std::array<std::unique_ptr<Tensor>, 4> input_tensors;
  for (int i = 0; i < 4; i++)
  {
    input_tensors[i] = std::move(predictor->GetInput(i);
    input_tensors[i]->Resize(INPUT_SHAPE);
    auto* input_data = input_tensor[i]->mutable_data<int64_t>();
    for (int i = 0; i < ShapeProduction(nput_tensor[i]->shape()); ++i) {
      input_data[i] = 1;
    }
  }
  // std::unique_ptr<Tensor> input_tensor_0(std::move(predictor->GetInput(0)));
  // std::unique_ptr<Tensor> input_tensor_0(std::move(predictor->GetInput(1)));
  // std::unique_ptr<Tensor> input_tensor_0(std::move(predictor->GetInput(2)));
  // std::unique_ptr<Tensor> input_tensor_0(std::move(predictor->GetInput(3)));
  // input_tensor_0->Resize(INPUT_SHAPE);
  // auto* input_data = input_tensor->mutable_data<float>();
  // for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
  //   input_data[i] = 1;
  // }
  // 2. Warmup Run
  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }
  // 3. Repeat Run
  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor->Run();
  }
  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: MobileNet_SSD, warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";
  // 4. Get results
  auto output_tensor = predictor->GetOutput(0);
  const auto* output_data = output_tensor->data<float>();

  // // 5. Check output
  // CHECK_EQ(ShapeProduction(output_tensor->shape()), 1000);
  
  // const int step = 50;
  // const double eps = 0.1;
  // for (int i = 0; i < ref.size(); ++i) {
  //   for (int j = 0; j < ref[i].size(); ++j) {
  //     auto result = output_data[j * step + (output_tensor->shape()[1] * i)];
  //     auto diff = std::fabs(result - ref[i][j]);
  //     // VLOG(3) << "expected[" << i <<"][" << j <<"] = " << ref[i][j]; 
  //     // VLOG(3) << "results[" << i <<"][" << j <<"] = " << result;
  //     // VLOG(3) << "diff[" << i <<"][" << j <<"] = " << diff;
  //     CHECK_LT(diff, eps) << "diff is not less than eps, diff is: " << diff << ", eps is: " << eps;
  //   }
  // }

  // 6. Print output
  printf("output_data size: %du\n", ShapeProduction(output_tensor->shape()));
  printf("output_data data: %du\n", output_data[0]);
  // std::vector<RESULT> results = postprocess(output_data, ShapeProduction(output_tensor->shape()));
  // printf("results: %du\n", results.size());
  // for (int i = 0; i < results.size(); i++) {
  //   printf("Top%d: %d - %f\n", i, results[i].class_id, results[i].score);
  // }
}

void RunModel(const std::string model_dir) {
  // 1. Create MobileConfig
  MobileConfig mobile_config;
  mobile_config.set_model_from_file(model_dir+".nb");
  mobile_config.set_threads(CPU_THREAD_NUM);
  mobile_config.set_power_mode(PowerMode::LITE_POWER_HIGH);
  mobile_config.set_subgraph_model_cache_dir(model_dir.substr(0, model_dir.find_last_of("/")));
  // mobile_config.set_huawei_ascend_device_id(1);

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  try {
    predictor = CreatePaddlePredictor<MobileConfig>(mobile_config);
    std::cout << "============== PaddlePredictor Version: " << predictor->GetVersion() << " ==============" << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(mobile config)." << std::endl;
  }
  // 3. Run model
  process(predictor);
}

void SaveModel(const std::string model_dir, const int model_type) {
  // 1. Create CxxConfig
  CxxConfig cxx_config;
  if (model_type) { // combined model
    cxx_config.set_model_file(model_dir + "/model");
    cxx_config.set_param_file(model_dir + "/params");
  } else {
    cxx_config.set_model_dir(model_dir);
  }
  cxx_config.set_threads(CPU_THREAD_NUM);
  cxx_config.set_power_mode(PowerMode::LITE_POWER_HIGH);
  std::vector<Place> valid_places({
      Place{TARGET(kHuaweiAscendNPU), PRECISION(kFloat)},
      Place{TARGET(kX86), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });
  cxx_config.set_valid_places(valid_places);
  cxx_config.set_subgraph_model_cache_dir(model_dir.substr(0, model_dir.find_last_of("/")));
  // cxx_config.set_device_id(1);

  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  try {
    predictor = CreatePaddlePredictor<CxxConfig>(cxx_config);
    std::cout << "============== PaddlePredictor Version: " << predictor->GetVersion() << " ==============" << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }

  // 3. Run model
  process(predictor);

  // 4. Save kNaiveBuffer model
  std::string save_optimized_model_file = model_dir + ".nb";
  int ret = system(string_format("rm -rf %s", save_optimized_model_file.c_str()).c_str());
  if (ret == 0) {
    LOG(INFO) << "Delete old optimized model " << save_optimized_model_file;
  }
  predictor->SaveOptimizedModel(model_dir, LiteModelType::kNaiveBuffer, /*record_info*/true);
  std::cout << "Save optimized native buffer model to " << (model_dir+".nb") << std::endl;

  // // 5. Save kProtobuf model
  // std::string save_optimized_model_dir = model_dir + "_opt";
  // int ret = system(string_format("rm -rf %s", save_optimized_model_dir.c_str()).c_str());
  // if (ret == 0) {
  //   LOG(INFO) << "Delete old optimized model " << save_optimized_model_dir;
  // }
  // predictor->SaveOptimizedModel(save_optimized_model_dir, LiteModelType::kProtobuf, /*record_info*/true);
  // std::cout << "Save optimized protobuf model to " << (model_dir+"_opt") << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << " model_dir\n";
    exit(1);
  }
  std::string model_dir = argv[1];
  int model_type = atoi(argv[2]); // 0 for uncombined, 1 for combined model

#ifdef USE_FULL_API
  SaveModel(model_dir, model_type);
#endif
  //RunModel(model_dir);
  return 0;
}