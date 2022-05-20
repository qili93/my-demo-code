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
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

int WARMUP_COUNT = 5;
int REPEAT_COUNT = 10;
const int CPU_THREAD_NUM = 1;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
    paddle::lite_api::PowerMode::LITE_POWER_HIGH;

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

template <typename T> void GetValueFromStream(std::stringstream *ss, T *t) {
  (*ss) >> (*t);
}

template <>
void GetValueFromStream<std::string>(std::stringstream *ss, std::string *t) {
  *t = ss->str();
}

template <typename T>
void Split(const std::string &line, char sep, std::vector<T> *v) {
  std::stringstream ss;
  T t;
  for (auto c : line) {
    if (c != sep) {
      ss << c;
    } else {
      GetValueFromStream<T>(&ss, &t);
      v->push_back(std::move(t));
      ss.str({});
      ss.clear();
    }
  }

  if (!ss.str().empty()) {
    GetValueFromStream<T>(&ss, &t);
    v->push_back(std::move(t));
    ss.str({});
    ss.clear();
  }
}

template <typename T>
bool ParseTensor(const std::vector<T> &data, const std::vector<int64_t> &shape,
                 paddle::lite_api::Tensor *tensor,
                 const std::vector<std::vector<uint64_t>> &lod = {}) {
  tensor->Resize(shape);
  tensor->SetLoD(lod);
  std::copy(data.begin(), data.end(), tensor->mutable_data<T>());
  return true;
}

bool ParseLine(
    const std::string &line, int max_seq_len, int max_out_len, int bos_idx,
    int eos_idx, int n_head,
    const std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor) {
  // src_word
  auto src_word = predictor->GetInput(0);
  std::vector<int64_t> src_word_data;
  Split(line, ' ', &src_word_data);
  int seq_len = src_word_data.size();
  for (int i = seq_len; i < max_seq_len; i++) {
    src_word_data.push_back(eos_idx);
  }
  std::vector<int64_t> src_word_shape{
      1, static_cast<int64_t>(src_word_data.size())};
  ParseTensor<int64_t>(src_word_data, src_word_shape, src_word.get());
  // src_pos
  auto src_pos = predictor->GetInput(1);
  std::vector<int64_t> src_pos_data(src_word_data.size());
  std::iota(src_pos_data.begin(), src_pos_data.end(), 0);
  std::vector<int64_t> src_pos_shape{1,
                                     static_cast<int64_t>(src_pos_data.size())};
  ParseTensor<int64_t>(src_pos_data, src_pos_shape, src_pos.get());
  // src_slf_attn_bias
  auto src_slf_attn_bias = predictor->GetInput(2);
  std::vector<float> src_slf_attn_bias_data(1 * n_head * src_word_data.size() *
                                            src_word_data.size());
  int offset = 0;
  for (int j = 0; j < 1 * n_head * src_word_data.size(); j++) {
    for (int i = 0; i < seq_len; i++) {
      src_slf_attn_bias_data[offset++] = 0.0f;
    }
    for (int i = seq_len; i < src_word_data.size(); i++) {
      src_slf_attn_bias_data[offset++] = -1e9f;
    }
  }
  std::vector<int64_t> src_slf_attn_bias_shape{
      1, n_head, static_cast<int64_t>(src_word_data.size()),
      static_cast<int64_t>(src_word_data.size())};
  ParseTensor<float>(src_slf_attn_bias_data, src_slf_attn_bias_shape,
                     src_slf_attn_bias.get());
  // trg_word
  auto trg_word = predictor->GetInput(3);
  std::vector<int64_t> trg_word_data(2, 0);
  std::vector<int64_t> trg_word_shape{2, 1};
  std::vector<uint64_t> lod_level_0{0, 2};
  std::vector<uint64_t> lod_level_1{0, 1, 2};
  std::vector<std::vector<uint64_t>> trg_word_lod(2);
  trg_word_lod[0] = lod_level_0;
  trg_word_lod[1] = lod_level_1;
  ParseTensor<int64_t>(trg_word_data, trg_word_shape, trg_word.get(),
                       trg_word_lod);
  // init_score
  auto init_score = predictor->GetInput(4);
  std::vector<float> init_score_data(2);
  init_score_data[0] = 0;
  init_score_data[1] = -1e9f;
  std::vector<int64_t> init_score_shape{2, 1};
  std::vector<std::vector<uint64_t>> init_score_lod(trg_word_lod);
  ParseTensor<float>(init_score_data, init_score_shape, init_score.get(),
                     init_score_lod);
  // init_idx
  auto init_idx = predictor->GetInput(5);
  std::vector<int32_t> init_idx_data(2, 0);
  std::vector<int64_t> init_idx_shape{2};
  ParseTensor<int32_t>(init_idx_data, init_idx_shape, init_idx.get());
  // trg_slf_attn_bias
  auto trg_slf_attn_bias = predictor->GetInput(6);
  std::vector<float> trg_slf_attn_bias_data(max_out_len * n_head * 1 *
                                            max_out_len);
  offset = 0;
  for (int k = 0; k < max_out_len; k++) {
    for (int j = 0; j < n_head; j++) {
      for (int i = 0; i < max_out_len; i++) {
        trg_slf_attn_bias_data[offset++] = (i <= k) ? 0.0f : -1e9f;
      }
    }
  }
  std::vector<int64_t> trg_slf_attn_bias_shape{max_out_len, n_head, 1,
                                               max_out_len};
  ParseTensor<float>(trg_slf_attn_bias_data, trg_slf_attn_bias_shape,
                     trg_slf_attn_bias.get());
  // trg_src_attn_bias
  auto trg_src_attn_bias = predictor->GetInput(7);
  std::vector<float> trg_src_attn_bias_data(1 * n_head * 1 *
                                            src_word_data.size());
  offset = 0;
  for (int j = 0; j < 1 * n_head * 1; j++) {
    for (int i = 0; i < seq_len; i++) {
      trg_src_attn_bias_data[offset++] = 0.0f;
    }
    for (int i = seq_len; i < src_word_data.size(); i++) {
      trg_src_attn_bias_data[offset++] = -1e9f;
    }
  }
  std::vector<int64_t> trg_src_attn_bias_shape{
      1, n_head, 1, static_cast<int64_t>(src_word_data.size())};
  ParseTensor<float>(trg_src_attn_bias_data, trg_src_attn_bias_shape,
                     trg_src_attn_bias.get());
  // kv_padding_selection
  auto kv_padding_selection = predictor->GetInput(8);
  std::vector<float> kv_padding_selection_data(max_out_len * n_head *
                                               max_out_len * 1);
  offset = 0;
  for (int k = 0; k < max_out_len; k++) {
    for (int j = 0; j < n_head; j++) {
      for (int i = 0; i < max_out_len; i++) {
        kv_padding_selection_data[offset++] = (i == k) ? 1.0f : 0.0f;
      }
    }
  }
  std::vector<int64_t> kv_padding_selection_shape{max_out_len, n_head,
                                                  max_out_len, 1};
  ParseTensor<float>(kv_padding_selection_data, kv_padding_selection_shape,
                     kv_padding_selection.get());
  return true;
}

std::vector<std::string> ParseInput(const std::string &path) {
  std::vector<std::string> lines;
  std::ifstream fin(path.c_str());
  if (fin.is_open()) {
    std::string line;
    while (std::getline(fin, line)) {
      lines.push_back(line);
    }
  }
  return lines;
}

void ParseOutput(
    const std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor,
    std::vector<int64_t> *ids, std::vector<float> *scores, int64_t bos = 0,
    int64_t eos = 1) {
  auto seq_ids = predictor->GetOutput(0);
  auto seq_scores = predictor->GetOutput(1);
  auto lod = seq_ids->lod();
  std::cout << "[";
  for (size_t i = 0; i < lod[0].size() - 1; i++) {
    size_t start = lod[0][i];
    size_t end = lod[0][i + 1];
    for (size_t j = 0; j < end - start; j++) {
      size_t sub_start = lod[1][start + j];
      size_t sub_end = lod[1][start + j + 1];
      auto seq_ids_data = seq_ids->data<int64_t>();
      std::cout << "[";
      for (size_t k = sub_start + 1; k < sub_end && seq_ids_data[k] != eos;
           k++) {
        ids->push_back(seq_ids_data[k]);
        std::cout << seq_ids_data[k] << " ";
      }
      std::cout << "-> ";
      auto last_seq_score = std::exp(-seq_scores->data<float>()[sub_end - 1]);
      scores->push_back(last_seq_score);
      std::cout << last_seq_score << "]";
    }
  }
  std::cout << "]" << std::endl;
}

void PrintOutput(
    const std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor) {
  auto seq_ids = predictor->GetOutput(0);
  auto seq_scores = predictor->GetOutput(1);
  auto seq_ids_shape = seq_ids->shape();
  auto seq_ids_size =
      std::accumulate(seq_ids_shape.begin(), seq_ids_shape.end(), 1,
                      std::multiplies<int64_t>());
  auto seq_ids_data = seq_ids->data<int64_t>();
  std::cout << "ids: [";
  for (int i = 0; i < seq_ids_size; i++) {
    std::cout << seq_ids_data[i] << " ";
  }
  std::cout << "]" << std::endl;
  auto seq_scores_shape = seq_scores->shape();
  auto seq_scores_size =
      std::accumulate(seq_scores_shape.begin(), seq_scores_shape.end(), 1,
                      std::multiplies<int64_t>());
  auto seq_scores_data = seq_scores->data<float>();
  std::cout << "scores: [";
  for (int i = 0; i < seq_scores_size; i++) {
    std::cout << seq_scores_data[i] << " ";
  }
  std::cout << "]" << std::endl;
}

int main(int argc, char **argv) {
  std::string model_dir = argv[1];
  int model_type = atoi(argv[2]);
  std::string test_file_path = argv[3];
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;

  const int bos_idx = 0;
  const int eos_idx = 1;
  const int n_head = 8;
  const int max_out_len = 8;
  int max_seq_len = 16;
  std::vector<int64_t> ids;
  std::vector<float> scores;
  double start, duration;

  auto test_lines = ParseInput(test_file_path);

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
       paddle::lite_api::Place{TARGET(kARM), PRECISION(kInt64)},
       paddle::lite_api::Place{TARGET(kNPU), PRECISION(kFloat)},
       paddle::lite_api::Place{TARGET(kNPU), PRECISION(kInt64)}});
  cxx_config.set_subgraph_model_cache_dir(
      model_dir.substr(0, model_dir.find_last_of("/")));
  try {
    predictor = paddle::lite_api::CreatePaddlePredictor(cxx_config);
    ParseLine(test_lines[0], max_seq_len, max_out_len, bos_idx, eos_idx, n_head,
              predictor);
    predictor->Run();
    ParseOutput(predictor, &ids, &scores);
    PrintOutput(predictor);
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
    start = GetCurrentUS();
    predictor = paddle::lite_api::CreatePaddlePredictor(mobile_config);
    duration = (GetCurrentUS() - start) / 1000.0f;
    std::cout << "load cost:" << duration << " ms" << std::endl;

    double total_cost = 0;
    double min_cost = 1e6;
    double max_cost = 0;
    std::string max_line = "";
    std::string min_line = "";
    for (int i = 0; i < 5 /*test_lines.size()*/; i++) {
      std::cout << i << std::endl;
      // no padding
      /*
      max_seq_len = -1;
      ParseLine(test_lines[i], max_seq_len, max_out_len, bos_idx, eos_idx,
      n_head, predictor); start = GetCurrentUS(); predictor->Run(); duration =
      (GetCurrentUS() - start) / 1000.0f; std::cout << "cost(no padding): " <<
      duration << " ms" << std::endl; ParseOutput(predictor, &ids, &scores);
      PrintOutput(predictor);
      */
      // padding
      max_seq_len = 16;
      ParseLine(test_lines[0], max_seq_len, max_out_len, bos_idx, eos_idx,
                n_head, predictor);
      start = GetCurrentUS();
      predictor->Run();
      duration = (GetCurrentUS() - start) / 1000.0f;
      std::cout << "cost(padding): " << duration << " ms" << std::endl;
      ParseOutput(predictor, &ids, &scores);
      PrintOutput(predictor);
      // statistic
      total_cost += duration;
      if (duration > max_cost) {
        max_cost = duration;
        max_line = test_lines[i];
      }
      if (duration < min_cost) {
        min_cost = duration;
        min_line = test_lines[i];
      }
      std::cout << "avg cost: " << total_cost / (i + 1) << "ms" << std::endl
                << "min cost: " << min_cost << "ms when data is '" << min_line
                << "'" << std::endl
                << "max cost: " << max_cost << "ms when data is '" << max_line
                << "'" << std::endl;
    }
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(mobile config)."
              << std::endl;
  }
  return 0;
}
