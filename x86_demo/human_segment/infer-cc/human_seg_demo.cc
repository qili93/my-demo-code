#include <gflags/gflags.h>
#include <glog/logging.h>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <memory>
#include <sys/time.h>
#include <paddle_inference_api.h>

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
const int CPU_THREAD_NUM = 1;

// // MODEL_NAME=align150-fp32
// const std::vector<int> INPUT_SHAPE = {1, 3, 128, 128};

// // MODEL_NAME=angle-fp32
// const std::vector<int> INPUT_SHAPE = {1, 3, 64, 64};

// // MODEL_NAME=detect_rgb-fp16
// const std::vector<int> INPUT_SHAPE = {1, 3, 320, 240};

// // MODEL_NAME=detect_rgb-int8
// const std::vector<int> INPUT_SHAPE = {1, 3, 320, 240};

// // MODEL_NAME=eyes_position-fp32
// const std::vector<int> INPUT_SHAPE = {1, 3, 32, 32};

// // MODEL_NAME=iris_position-fp32
// const std::vector<int> INPUT_SHAPE = {1, 3, 24, 24};

// // MODEL_NAME=mouth_position-fp32
// const std::vector<int> INPUT_SHAPE = {1, 3, 48, 48};

// MODEL_NAME=seg-model-int8
const std::vector<int> INPUT_SHAPE = {1, 4, 192, 192};

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

double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

int ShapeProduction(const std::vector<int>& shape) {
  int res = 1;
  for (auto i : shape) res *= i;
  return res;
}

void RunModel(const std::string model_dir, const int model_type) {
  // 1. Create AnalysisConfig
  paddle::AnalysisConfig config;
  if (model_type) {
    config.SetModel(model_dir + "/__model__",
                    model_dir + "/__params__");
  } else {
    config.SetModel(model_dir);
  }
  // use ZeroCopyTensor, Must be set to false
  config.SwitchUseFeedFetchOps(false);
  config.SetCpuMathLibraryNumThreads(CPU_THREAD_NUM);
  // turn off for int8 model
  config.SwitchIrOptim(false);

  // 2. Create PaddlePredictor by AnalysisConfig
  auto predictor = paddle::CreatePaddlePredictor(config);
  if (predictor == nullptr) {
    std::cout << "An internal error occurred in PaddlePredictor(AnalysisConfig)." << std::endl;
  }
  int nums = ShapeProduction(INPUT_SHAPE);
  float* input = new float[nums];
  for (int i = 0; i < nums; ++i) input[i] = 1.f;

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
  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << model_dir;
  LOG(INFO) << "Warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << (end_time - start_time) / FLAGS_repeats / 1000.0
            << " ms in average.";
  // 7. Get output
  std::vector<float> output_data;
  auto output_names = predictor->GetOutputNames();
  auto output_tensor = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_tensor->shape();
  int output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  output_data.resize(output_size);
  output_tensor->copy_to_cpu(output_data.data());

  // 8. free memory
  delete[] input;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << " model_dir model_type\n";
    exit(1);
  }
  std::string model_dir = argv[1];
    // 0 for uncombined, 1 for combined model
  int model_type = atoi(argv[2]);

  LOG(INFO) << "model_type=" << model_type;

  RunModel(model_dir, model_type);
  return 0;
}
