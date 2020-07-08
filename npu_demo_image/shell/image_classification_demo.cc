// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle_api.h"
#include <fstream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

int WARMUP_COUNT = 5;
int REPEAT_COUNT = 10;
const int CPU_THREAD_NUM = 1;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
    paddle::lite_api::PowerMode::LITE_POWER_HIGH;
const std::vector<int64_t> INPUT_SHAPE = {1, 3, 224, 224};
const std::vector<float> INPUT_MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> INPUT_STD = {0.229f, 0.224f, 0.225f};

struct RESULT {
  std::string class_name;
  int class_id;
  float score;
};

inline int64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

std::vector<std::string> load_labels(const std::string &path) {
  std::ifstream file;
  std::vector<std::string> labels;
  file.open(path);
  while (file) {
    std::string line;
    std::getline(file, line);
    std::string::size_type pos = line.find(" ");
    if (pos != std::string::npos) {
      line = line.substr(pos);
    }
    labels.push_back(line);
  }
  file.clear();
  file.close();
  return labels;
}

void preprocess(cv::Mat &input_image, const std::vector<float> &input_mean,
                const std::vector<float> &input_std, int input_width,
                int input_height, float *input_data) {
  cv::Mat resize_image;
  cv::resize(input_image, resize_image, cv::Size(input_width, input_height), 0,
             0);
  if (resize_image.channels() == 4) {
    cv::cvtColor(resize_image, resize_image, cv::COLOR_BGRA2RGB);
  }
  cv::Mat norm_image;
  resize_image.convertTo(norm_image, CV_32FC3, 1 / 255.f);
  // NHWC->NCHW
  int image_size = input_height * input_width;
  const float *image_data = reinterpret_cast<const float *>(norm_image.data);
  float *input_data_c0 = input_data;
  float *input_data_c1 = input_data + image_size;
  float *input_data_c2 = input_data + image_size * 2;
  int i = 0;
  float32x4_t vmean0 = vdupq_n_f32(input_mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(input_mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(input_mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(1.0f / input_std[0]);
  float32x4_t vscale1 = vdupq_n_f32(1.0f / input_std[1]);
  float32x4_t vscale2 = vdupq_n_f32(1.0f / input_std[2]);
  for (; i < image_size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(image_data);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(input_data_c0, vs0);
    vst1q_f32(input_data_c1, vs1);
    vst1q_f32(input_data_c2, vs2);
    image_data += 12;
    input_data_c0 += 4;
    input_data_c1 += 4;
    input_data_c2 += 4;
  }
  for (; i < image_size; i++) {
    *(input_data_c0++) = (*(image_data++) - input_mean[0]) / input_std[0];
    *(input_data_c1++) = (*(image_data++) - input_mean[1]) / input_std[1];
    *(input_data_c2++) = (*(image_data++) - input_mean[2]) / input_std[2];
  }
}

bool topk_compare_func(std::pair<float, int> a, std::pair<float, int> b) {
  return (a.first > b.first);
}

std::vector<RESULT> postprocess(const float *output_data, int64_t output_size,
                                const std::vector<std::string> &word_labels) {
  const int TOPK = 3;
  std::vector<std::pair<float, int>> vec;
  for (int i = 0; i < output_size; i++) {
    vec.push_back(std::make_pair(output_data[i], i));
  }
  std::partial_sort(vec.begin(), vec.begin() + TOPK, vec.end(),
                    topk_compare_func);
  std::vector<RESULT> results(TOPK);
  for (int i = 0; i < TOPK; i++) {
    results[i].score = vec[i].first;
    results[i].class_id = vec[i].second;
    results[i].class_name = "Unknown";
    if (results[i].class_id >= 0 && results[i].class_id < word_labels.size()) {
      results[i].class_name = word_labels[results[i].class_id];
    }
  }
  return results;
}

void process(cv::Mat &input_image, std::vector<std::string> &word_labels,
             std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor) {
  // Preprocess image and fill the data of input tensor
  std::unique_ptr<paddle::lite_api::Tensor> input_tensor(
      std::move(predictor->GetInput(0)));
  input_tensor->Resize(INPUT_SHAPE);
  int input_width = INPUT_SHAPE[3];
  int input_height = INPUT_SHAPE[2];
  auto *input_data = input_tensor->mutable_data<float>();
  double preprocess_start_time = get_current_us();
  preprocess(input_image, INPUT_MEAN, INPUT_STD, input_width, input_height,
             input_data);
  double preprocess_end_time = get_current_us();
  double preprocess_time =
      (preprocess_end_time - preprocess_start_time) / 1000.0f;

  double prediction_time;
  // Run predictor
  // warm up to skip the first inference and get more stable time, remove it in
  // actual products
  for (int i = 0; i < WARMUP_COUNT; i++) {
    predictor->Run();
  }
  // repeat to obtain the average time, set REPEAT_COUNT=1 in actual products
  double max_time_cost = 0.0f;
  double min_time_cost = std::numeric_limits<float>::max();
  double total_time_cost = 0.0f;
  for (int i = 0; i < REPEAT_COUNT; i++) {
    auto start = get_current_us();
    predictor->Run();
    auto end = get_current_us();
    double cur_time_cost = (end - start) / 1000.0f;
    if (cur_time_cost > max_time_cost) {
      max_time_cost = cur_time_cost;
    }
    if (cur_time_cost < min_time_cost) {
      min_time_cost = cur_time_cost;
    }
    total_time_cost += cur_time_cost;
    prediction_time = total_time_cost / REPEAT_COUNT;
    printf("iter %d cost: %f ms\n", i, cur_time_cost);
  }
  printf("warmup: %d repeat: %d, average: %f ms, max: %f ms, min: %f ms\n",
         WARMUP_COUNT, REPEAT_COUNT, prediction_time, max_time_cost,
         min_time_cost);

  // Get the data of output tensor and postprocess to output detected objects
  std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  // const float *output_data = output_tensor->mutable_data<float>();
  const float *output_data = output_tensor->data<float>();
  int64_t output_size = 1;
  for (auto dim : output_tensor->shape()) {
    output_size *= dim;
  }
  double postprocess_start_time = get_current_us();
  std::vector<RESULT> results =
      postprocess(output_data, output_size, word_labels);
  double postprocess_end_time = get_current_us();
  double postprocess_time =
      (postprocess_end_time - postprocess_start_time) / 1000.0f;

  printf("results: %d\n", results.size());
  for (int i = 0; i < results.size(); i++) {
    printf("Top%d %s - %f\n", i, results[i].class_name.c_str(),
           results[i].score);
  }
  printf("Preprocess time: %f ms\n", preprocess_time);
  printf("Prediction time: %f ms\n", prediction_time);
  printf("Postprocess time: %f ms\n\n", postprocess_time);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("Usage: \n"
           "./image_classification_demo model_dir model_type label_path "
           "image_path");
    return -1;
  }
  std::string model_dir = argv[1];
  int model_type = atoi(argv[2]);
  std::string label_path = argv[3];
  std::string image_path = argv[4];
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;

  // Load label and image file
  std::vector<std::string> word_labels = load_labels(label_path);
  cv::Mat input_image = cv::imread(image_path, 1);

#ifdef USE_FULL_API
  // Run inference by using full api with CxxConfig
  paddle::lite_api::CxxConfig cxx_config;
  if (model_type) { // combined model
    cxx_config.set_model_file(model_dir + "/model");
    cxx_config.set_param_file(model_dir + "/params");
  } else {
    cxx_config.set_model_dir(model_dir);
  }
  cxx_config.set_threads(CPU_THREAD_NUM);
  cxx_config.set_power_mode(CPU_POWER_MODE);
  cxx_config.set_valid_places(
      {paddle::lite_api::Place{TARGET(kARM), PRECISION(kFloat)},
       /*paddle::lite_api::Place{TARGET(kNPU), PRECISION(kFloat)}*/});
  cxx_config.set_subgraph_model_cache_dir(
      model_dir.substr(0, model_dir.find_last_of("/")));
  try {
    predictor = paddle::lite_api::CreatePaddlePredictor(cxx_config);
    process(input_image, word_labels, predictor);
    predictor->SaveOptimizedModel(
        model_dir, paddle::lite_api::LiteModelType::kNaiveBuffer);
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)."
              << std::endl;
  }
#endif

  // Run inference by using light api with MobileConfig
  paddle::lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(model_dir + ".nb");
  mobile_config.set_threads(CPU_THREAD_NUM);
  mobile_config.set_power_mode(CPU_POWER_MODE);
  try {
    predictor = paddle::lite_api::CreatePaddlePredictor(mobile_config);
    process(input_image, word_labels, predictor);
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(mobile config)."
              << std::endl;
  }

  return 0;
}
