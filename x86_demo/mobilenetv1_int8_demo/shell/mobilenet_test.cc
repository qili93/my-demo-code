#include <gflags/gflags.h>
#include <glog/logging.h>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <memory>
#include <sys/time.h>
#include "paddle/include/paddle_inference_api.h"
#include "paddle/include/paddle_analysis_config.h"
#include "paddle/include/paddle_mkldnn_quantizer_config.h"

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
const int CPU_THREAD_NUM = 1;

// Mobilenet_v1
const std::vector<int> IMAGE_SHAPE = {1, 3, 224, 224};
const std::vector<int> LABEL_SHAPE = {1, 1};

namespace paddle {

using paddle::AnalysisConfig;

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

std::shared_ptr<std::vector<PaddleTensor>> GetDummyWarmupData() {
  std::vector<size_t> accum_lod;
  accum_lod.push_back(1UL);

  PaddleTensor images;
  images.name = "image";
  images.shape = IMAGE_SHAPE;
  int image_size = std::accumulate(IMAGE_SHAPE.begin(), IMAGE_SHAPE.end(), 1, std::multiplies<int>());
  images.dtype = PaddleDType::FLOAT32;
  images.data.Resize(sizeof(float) * image_size);
  float * image_data = static_cast<float *>(images.data.data());
  for (int i = 0; i < image_size; ++i) {
    image_data[i] = 1.f;
  }

  PaddleTensor labels;
  labels.name = "label";
  labels.shape = LABEL_SHAPE;
  int label_size = std::accumulate(LABEL_SHAPE.begin(), LABEL_SHAPE.end(), 1, std::multiplies<int>());
  labels.dtype = PaddleDType::INT64;
  labels.data.Resize(sizeof(int64_t) * label_size);
  // int64_t * labels_data = static_cast<int64_t *>(labels.data.data());
  labels.lod.push_back(accum_lod);
  // for (int i = 0; i < label_size; ++i) {
  //   labels_data[i] = 1;
  // }

  auto warmup_data = std::make_shared<std::vector<PaddleTensor>>(2);
  (*warmup_data)[0] = std::move(images);
  (*warmup_data)[1] = std::move(labels);
  return warmup_data;
}

void RunModel(const std::string model_dir, const int model_type) {
  // 1. Create AnalysisConfig
  AnalysisConfig config;

  if (model_type) { // combined model
    config.SetModel(model_dir + "/model", model_dir + "/params");
  } else {
    config.SetModel(model_dir);
  }
  config.SetModel(model_dir);
  config.DisableGpu();
  config.SwitchIrOptim();
  config.SwitchSpecifyInputNames();
  config.SetCpuMathLibraryNumThreads(1);
  config.EnableMKLDNN();
  // We use ZeroCopyTensor here, so we set config->SwitchUseFeedFetchOps(false)
  config.SwitchUseFeedFetchOps(false);
  // Enable int8
  config.EnableMkldnnQuantizer();
  // create mkldnn_quantizer_config
  std::shared_ptr<std::vector<PaddleTensor>> warmup_data = GetDummyWarmupData();
  config.mkldnn_quantizer_config()->SetWarmupData(warmup_data);
  config.mkldnn_quantizer_config()->SetWarmupBatchSize(1);

  // 2. Create PaddlePredictor by AnalysisConfig
  auto predictor = CreatePaddlePredictor(config);
  if (!predictor) {
    std::cout << "An internal error occurred in PaddlePredictor(AnalysisConfig)." << std::endl;
  }
  // 3. Prepare input data
  auto input_names = predictor->GetInputNames();
  auto image_tensor = predictor->GetInputTensor(input_names[0]);
  auto label_tensor = predictor->GetInputTensor(input_names[1]);
  image_tensor->Reshape(IMAGE_SHAPE);
  label_tensor->Reshape(LABEL_SHAPE);
  // set image data
  int image_size = std::accumulate(IMAGE_SHAPE.begin(), IMAGE_SHAPE.end(), 1, std::multiplies<int>());
  float *image_data = image_tensor->mutable_data<float>(PaddlePlace::kCPU);
  for (int i = 0; i < image_size; ++i) {
    image_data[i] = 1.f;
  }
  // set label data
  int label_size = std::accumulate(LABEL_SHAPE.begin(), LABEL_SHAPE.end(), 1, std::multiplies<int>());
  int64_t *label_data = label_tensor->mutable_data<int64_t>(PaddlePlace::kCPU);
  for (int i = 0; i < label_size; ++i) {
    label_data[i] = 1;
  }

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
  // 8. Print output
  // std::vector<RESULT> results = postprocess(output_data.data(), output_size);
  // printf("results: %lu\n", results.size());
  // for (int i = 0; i < results.size(); i++) {
  //   printf("Top%d: %d - %f\n", i, results[i].class_id, results[i].score);
  // }
}
}  // namespace paddle

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << " model_dir model_type\n";
    exit(1);
  }
  std::string model_dir = argv[1];
    // 0 for uncombined, 1 for combined model
  int model_type = atoi(argv[2]);

  paddle::RunModel(model_dir, model_type);
  return 0;
}
