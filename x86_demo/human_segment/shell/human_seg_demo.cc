#include <iostream>
#include <algorithm>
#include "paddle_api.h"
#include "logging.h"

#if defined(_WIN32)
#include<sys/timeb.h>
#endif

#ifdef WIN32
#define OS_SEP '\\'
#else
#define OS_SEP '/'
#endif

using namespace paddle::lite_api;  // NOLINT

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
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

#if !defined(_WIN32)
double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}
#else
double GetCurrentUS() {
  struct timeb cur_time;
  ftime(&cur_time);
  return (cur_time.time * 1e+6) + cur_time.millitm * 1e+3;
}
#endif

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
  // std::vector<RESULT> results = postprocess(output_data, ShapeProduction(output_tensor->shape()));
  // printf("results: %du\n", results.size());
  // for (size_t i = 0; i < results.size(); i++) {
  //   printf("Top%d: %d - %f\n", i, results[i].class_id, results[i].score);
  // }
}

void RunModel(std::string model_path, const std::vector<int64_t> input_shape_vec) {
  // 1. Create MobileConfig
  MobileConfig mobile_config;
  mobile_config.set_model_from_file(model_path+".nb");
  mobile_config.set_threads(CPU_THREAD_NUM);
  mobile_config.set_power_mode(PowerMode::LITE_POWER_HIGH);
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor = nullptr;
  // 2. Create PaddlePredictor by MobileConfig
  try {
    predictor = CreatePaddlePredictor<MobileConfig>(mobile_config);
    std::cout << "==============MobileConfig Predictor Version: " << predictor->GetVersion() << " ==============" << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(mobile config)." << std::endl;
  }
  // 3. Run model
  process(predictor, input_shape_vec);
}

#ifdef USE_FULL_API
void SaveModel(std::string model_path, const int model_type, const std::vector<int64_t> input_shape_vec) {
  LOG(INFO) << "entering cxx_config";
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
  LOG(INFO) << "finish cxx_config";
  // cxx_config.set_subgraph_model_cache_dir(model_path.substr(0, model_path.find_last_of("/")));

  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<PaddlePredictor> predictor = nullptr;
  try {
    predictor = CreatePaddlePredictor<CxxConfig>(cxx_config);
    std::cout << "============== CxxConfig Predictor Version: " << predictor->GetVersion() << " ==============" << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }
  LOG(INFO) << "finish predictor";

  // 3. Run model
  //process(predictor, input_shape_vec);

  // 4. Save optimized model
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

  // set input shape based on model name
  std::vector<int64_t> input_shape_vec(4);
  if (model_name == "align150-fp32") {
    int64_t input_shape[] = {1, 3, 128, 128};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "angle-fp32") {
    int64_t input_shape[] = {1, 3, 64, 64};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "detect_rgb-fp32") {
    int64_t input_shape[] = {1, 3, 320, 240};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "detect_rgb-int8") {
    int64_t input_shape[] = {1, 3, 320, 240};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "eyes_position-fp32") {
    int64_t input_shape[] = {1, 3, 32, 32};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "iris_position-fp32") {
    int64_t input_shape[] = {1, 3, 24, 24};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "mouth_position-fp32") {
    int64_t input_shape[] = {1, 3, 48, 48};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "seg-model-int8") {
    int64_t input_shape[] = {1, 4, 192, 192};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "pc-seg-float-model") {
    int64_t input_shape[] = {1, 4, 192, 256};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else {
    LOG(ERROR) << "NOT supported model name!";
    return 0;
  }

  LOG(INFO) << "Model Name is <" << model_name << ">, Input Shape is {" 
    << input_shape_vec[0] << ", " << input_shape_vec[1] << ", " 
    << input_shape_vec[2] << ", " << input_shape_vec[3] << "}";

  std::string model_path = model_dir + OS_SEP + model_name;

  LOG(INFO) << "Model Path is <" << model_path << ">";

#ifdef USE_FULL_API
  SaveModel(model_path, model_type, input_shape_vec);
#endif

  //RunModel(model_path, input_shape_vec);

  return 0;
}


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
// const std::vector<int64_t> INPUT_SHAPE = {1, 4, 192, 256};