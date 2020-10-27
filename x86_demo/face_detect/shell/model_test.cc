#include <iostream>
#include <algorithm>
#include "paddle_api.h"
#include "logging.h"

using namespace paddle::lite_api;  // NOLINT

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
const int CPU_THREAD_NUM = 1;

// MODEL_NAME=face_detect_fp32
const std::vector<int64_t> INPUT_SHAPE = {1, 3, 320, 512};

void read_imgnp(const std::string raw_imgnp_path, const std::vector<int64_t> input_shape_vec, float * input_data) {
  std::ifstream raw_imgnp_file(raw_imgnp_path, std::ios::in | std::ios::binary);
  if (!raw_imgnp_file) {
    std::cout << "Failed to load raw rgb image file: " <<  raw_imgnp_path << std::endl;
    return;
  }
  int64_t raw_imgnp_size = ShapeProduction(input_shape_vec);
  raw_imgnp_file.read(reinterpret_cast<char *>(input_data), raw_imgnp_size * sizeof(float));
  raw_imgnp_file.close();
}

std::vector<RESULT> postprocess(const float *output_data, int64_t output_size) {
  const int TOPK = std::min(10, static_cast<int>(output_size));
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

void process(std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor, const std::string imgnp_path, const std::vector<int64_t> input_shape_vec) {
  // 1. Prepare input data
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize(input_shape_vec);
  auto* input_data = input_tensor->mutable_data<float>();
  read_imgnp(imgnp_path, input_shape_vec, input_data);

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
  // 5. Get all output
  int output_num = static_cast<int>(predictor->GetOutputNames().size());
  for (int i = 0; i < output_num; ++i) {
    std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(i)));
    const float *output_data = output_tensor->data<float>();
    LOG(INFO) << "Printing Output Index: <" << i << ">, shape is " << shape_to_string(output_tensor->shape());
    // std::vector<RESULT> results = postprocess(output_data, ShapeProduction(output_tensor->shape()));
    // for (size_t j = 0; j < results.size(); j++) {
    //   LOG(INFO) << "Top "<< j <<": " << results[j].class_id << " - " << results[j].score;
    // }
    LOG(INFO) << data_to_string<float>(output_data, ShapeProduction(output_tensor->shape()));
  }
}

void RunLiteModel(const std::string model_path, const std::string imgnp_path, const std::vector<int64_t> input_shape_vec) {
  // 1. Create MobileConfig
  auto start_time = GetCurrentUS();
  MobileConfig mobile_config;
  mobile_config.set_model_from_file(model_path+".nb");
  // Load model from buffer
  // std::string model_buffer = ReadFile(model_path+".nb");
  // mobile_config.set_model_from_buffer(model_buffer);
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
  auto end_time = GetCurrentUS();

  // 3. Run model
  process(predictor, imgnp_path, input_shape_vec);
  LOG(INFO) << "MobileConfig preprosss: " 
            << (end_time - start_time) / 1000.0
            << " ms.";
}

#ifdef USE_FULL_API
void RunFullModel(const std::string model_path, const std::vector<int64_t> input_shape_vec) {
  // 1. Create CxxConfig
  auto start_time = GetCurrentUS();
  CxxConfig cxx_config;
  cxx_config.set_model_file(model_path + "_opt/model");
  cxx_config.set_param_file(model_path + "_opt/params");
  cxx_config.set_valid_places({Place{TARGET(kX86), PRECISION(kFloat)},
                               Place{TARGET(kHost), PRECISION(kFloat)}});
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor = nullptr;
  // 2. Create PaddlePredictor by MobileConfig
  try {
    predictor = CreatePaddlePredictor<CxxConfig>(cxx_config);
    std::cout << "==============CxxConfig Predictor Version: " << predictor->GetVersion() << " ==============" << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }
  auto end_time = GetCurrentUS();
  // 3. Run model
  process(predictor, input_shape_vec);
  LOG(INFO) << "CXXConfig preprosss: " 
            << (end_time - start_time) / 1000.0
            << " ms.";
}

void SaveModel(const std::string model_path, const int model_type, const std::vector<int64_t> input_shape_vec) {
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
  // cxx_config.set_subgraph_model_cache_dir(model_path.substr(0, model_path.find_last_of("/")));

  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<PaddlePredictor> predictor = nullptr;
  try {
    predictor = CreatePaddlePredictor<CxxConfig>(cxx_config);
    std::cout << "============== CxxConfig Predictor Version: " << predictor->GetVersion() << " ==============" << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }

  // 3. Save optimized model
  predictor->SaveOptimizedModel(model_path, LiteModelType::kNaiveBuffer);
  std::cout << "Save optimized model to " << (model_path+".nb") << std::endl;

  predictor->SaveOptimizedModel(model_path+"_opt", LiteModelType::kProtobuf);
  std::cout << "Save optimized model to " << (model_path+"_opt") << std::endl;
}
#endif

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << "assets_dir model_name image_name\n";
    exit(1);
  }
  std::string assets_dir = argv[1];
  std::string model_name = argv[2];
  std::string image_name = argv[3];
  // 0 for uncombined, 1 for combined model
  // int model_type = atoi(argv[3]);
  int model_type = 1;

  // // set input shape based on model name
  std::vector<int64_t> input_shape_vec(4);


  int64_t input_shape[] = {1, 3, 320, 512};
  std::copy (input_shape, input_shape + 4, input_shape_vec.begin());

  LOG(INFO) << "Model Name is <" << model_name << ">, Input Shape is {" 
    << input_shape_vec[0] << ", " << input_shape_vec[1] << ", " 
    << input_shape_vec[2] << ", " << input_shape_vec[3] << "}";

  std::string model_path = assets_dir + "/models/" + model_name;
  std::string image_path = assets_dir + "/images/" + image_name;

  LOG(INFO) << "Model Path is <" << model_path << ">";
  LOG(INFO) << "Image Path is <" << image_path << ">";

#ifdef USE_FULL_API
  SaveModel(model_path, model_type, input_shape_vec);
  RunFullModel(model_path, input_shape_vec);
#endif

  RunLiteModel(model_path, image_path, input_shape_vec);

  return 0;
}
