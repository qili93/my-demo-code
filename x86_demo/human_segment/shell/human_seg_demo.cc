#include <iostream>
#include <algorithm>
#include "paddle_api.h"
#include "logging.h"

using namespace paddle::lite_api;  // NOLINT

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;

const int CPU_THREAD_NUM = 1;


// // MODEL_NAME=align150-fp32
// const std::vector<int64_t> INPUT_SHAPE = {1, 3, 128, 128};

// // MODEL_NAME=angle-fp32
// const std::vector<int64_t> INPUT_SHAPE = {1, 3, 64, 64};

// // MODEL_NAME=detect_rgb-fp32
// const std::vector<int64_t> INPUT_SHAPE = {1, 3, 320, 240};

// // MODEL_NAME=detect_rgb-int8
// const std::vector<int64_t> INPUT_SHAPE = {1, 3, 320, 240};

// // MODEL_NAME=eyes_position-fp32
// const std::vector<int64_t> INPUT_SHAPE = {1, 3, 32, 32};

// // MODEL_NAME=iris_position-fp32
// const std::vector<int64_t> INPUT_SHAPE = {1, 3, 24, 24};

// // MODEL_NAME=mouth_position-fp32
// const std::vector<int64_t> INPUT_SHAPE = {1, 3, 48, 48};

// // MODEL_NAME=seg-model-int8
// const std::vector<int64_t> INPUT_SHAPE = {1, 4, 192, 192};

// MODEL_NAME=pc-seg-float-model
const std::vector<int64_t> INPUT_SHAPE = {1, 4, 192, 256};


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

void process(std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor, const std::string model_name) {
  // 1. Prepare input data
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
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
  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << model_name;
  LOG(INFO) << "Warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << (end_time - start_time) / FLAGS_repeats / 1000.0
            << " ms in average.";

  // 5. Get output
  std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
  const float *output_data = output_tensor->data<float>();
  // 6. Print output
  // std::vector<RESULT> results = postprocess(output_data, ShapeProduction(output_tensor->shape()));
  // printf("results: %du\n", results.size());
  // for (size_t i = 0; i < results.size(); i++) {
  //   printf("Top%d: %d - %f\n", i, results[i].class_id, results[i].score);
  // }
}

void RunModel(std::string model_name) {
  // 1. Create MobileConfig
  MobileConfig mobile_config;
  mobile_config.set_model_from_file(model_name+".nb");
  mobile_config.set_threads(CPU_THREAD_NUM);
  mobile_config.set_power_mode(PowerMode::LITE_POWER_HIGH);
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor = nullptr;
  // 2. Create PaddlePredictor by MobileConfig
  try {
    predictor = CreatePaddlePredictor<MobileConfig>(mobile_config);
    std::cout << "============== PaddlePredictor Version: " << predictor->GetVersion() << " ==============" << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(mobile config)." << std::endl;
  }
  // 3. Run model
  process(predictor, model_name);
}

#ifdef USE_FULL_API
void SaveModel(std::string model_dir, const int model_type) {
  // 1. Create CxxConfig
  CxxConfig cxx_config;
  if (model_type) { // combined model
    cxx_config.set_model_file(model_dir + "/__model__");
    cxx_config.set_param_file(model_dir + "/__params__");
  } else {
    cxx_config.set_model_dir(model_dir);
  }
  cxx_config.set_valid_places({Place{TARGET(kX86), PRECISION(kFloat)},
                           Place{TARGET(kHost), PRECISION(kFloat)}});
  // cxx_config.set_subgraph_model_cache_dir(model_dir.substr(0, model_dir.find_last_of("/")));

  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<PaddlePredictor> predictor = nullptr;
  try {
    predictor = CreatePaddlePredictor<CxxConfig>(cxx_config);
    std::cout << "============== PaddlePredictor Version: " << predictor->GetVersion() << " ==============" << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }

  // 3. Run model
  process(predictor, model_dir);

  // 4. Save optimized model
  predictor->SaveOptimizedModel(model_dir, LiteModelType::kNaiveBuffer);
  // predictor->SaveOptimizedModel(model_dir+"_opt", LiteModelType::kProtobuf);
  std::cout << "Load model from " << model_dir << std::endl;
  std::cout << "Save optimized model to " << (model_dir+".nb") << std::endl;
}
#endif

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << " model_dir model_type\n";
    exit(1);
  }
  std::string model_dir = argv[1];
  // 0 for uncombined, 1 for combined model
  int model_type = atoi(argv[2]);

#ifdef USE_FULL_API
  SaveModel(model_dir, model_type);
#endif

  RunModel(model_dir);

  return 0;
}
