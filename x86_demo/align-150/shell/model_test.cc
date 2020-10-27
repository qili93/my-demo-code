#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <memory>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <paddle_api.h>

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
const int CPU_THREAD_NUM = 1;

const std::string IMAGE_FILE_NAME = "face-crop.jpg"; // {1, 3, 1050, 1682} NCHW
const std::string IMAGE_DATA_NAME = "face-crop.raw"; // {1, 3, 128, 128} # float32

// MODEL_NAME=align150-fp32
const std::vector<int64_t> INPUT_SHAPE = {1, 3, 128, 128};

template <typename T>
static std::string data_to_string(const T* data, const int64_t size) {
  std::stringstream ss;
  ss << "{";
  for (int64_t i = 0; i < size - 1; ++i) {
    ss << data[i] << ",";
  }
  ss << data[size - 1] << "}";
  return ss.str();
}

static std::string shape_to_string(const std::vector<int64_t>& shape) {
  std::stringstream ss;
  if (shape.empty()) {
    ss << "{}";
    return ss.str();
  }
  ss << "{";
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    ss << shape[i] << ",";
  }
  ss << shape[shape.size() - 1] << "}";
  return ss.str();
}

int64_t ShapeProduction(const std::vector<int64_t>& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

void read_imgnp(const std::string raw_imgnp_path, float * input_data) {
  std::ifstream raw_imgnp_file(raw_imgnp_path, std::ios::in | std::ios::binary);
  if (!raw_imgnp_file) {
    std::cout << "Failed to load raw image file: " <<  raw_imgnp_path << std::endl;
    return;
  }
  int64_t raw_imgnp_size = ShapeProduction(INPUT_SHAPE);
  raw_imgnp_file.read(reinterpret_cast<char *>(input_data), raw_imgnp_size * sizeof(float));
  raw_imgnp_file.close();
}

// save output to raw file
void write_file(const float * output_data, const std::vector<int64_t>& shape, const std::string output_file_path) {
  std::ofstream output_file(output_file_path, std::ios::out | std::ios::binary);
  if (!output_file) {
    std::cout << "Failed to open raw output file: " <<  output_file_path << std::endl;
    return;
  }
  int64_t output_size = ShapeProduction(shape);
  output_file.write(reinterpret_cast<const char *>(output_data), output_size * sizeof(float));
  output_file.close();
}

void process(std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor, const std::string image_path) {
  // 1. Prepare input data
  std::unique_ptr<paddle::lite_api::Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize(INPUT_SHAPE);
  auto* input_data = input_tensor->mutable_data<float>();
  for (size_t i = 0; i < ShapeProduction(INPUT_SHAPE); ++i) {
    input_data[i] = 1.0f;
  }
  // read_imgnp(image_path, input_data);

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
  std::cout << "================== Speed Report ===================" << std::endl;
  std::cout << "Warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << (end_time - start_time) / FLAGS_repeats / 1000.0
            << " ms in average." << std::endl;
  // 5. Get output 0

  std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(std::move(predictor->GetOutput(0)));
  const float *output_data = output_tensor->data<float>();
  const int64_t ouput_size = ShapeProduction(output_tensor->shape());
  std::cout << "Printing Output Index: <0>, shape is " << shape_to_string(output_tensor->shape()) << std::endl;
  write_file(output_data, output_tensor->shape(), OUTPUT_FILE_PATH);
  std::cout << "Printing Output Index: <0>, data is " << data_to_string(output_data, ouput_size) << std::endl;
  // 6. Get output 1
  std::unique_ptr<const paddle::lite_api::Tensor> output_tensor_score(std::move(predictor->GetOutput(1)));
  const float *output_data_score = output_tensor_score->data<float>();
  std::cout << "Printing Output Index: <1>, shape is " << shape_to_string(output_tensor_score->shape()) << std::endl;
  write_file(output_data_score, output_tensor_score->shape(), OUTPUT_SCORE_PATH);
}

void RunLiteModel(const std::string model_path, const std::string image_path) {
  // 1. Create MobileConfig
  auto start_time = GetCurrentUS();
  paddle::lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(model_path+".nb");
  // Load model from buffer
  // std::string model_buffer = ReadFile(model_path+".nb");
  // mobile_config.set_model_from_buffer(model_buffer);
  mobile_config.set_threads(CPU_THREAD_NUM);
  mobile_config.set_power_mode(paddle::lite_api::PowerMode::LITE_POWER_HIGH);
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  // 2. Create PaddlePredictor by MobileConfig
  try {
    predictor = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(mobile_config);
    std::cout << "==============MobileConfig Predictor Version: " << predictor->GetVersion() << " ==============" << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(mobile config)." << std::endl;
  }
  auto end_time = GetCurrentUS();

  // 3. Run model
  process(predictor, image_path);
  std::cout << "MobileConfig preprosss: " 
            << (end_time - start_time) / 1000.0
            << " ms." << std::endl;
}

#ifdef USE_FULL_API
void RunFullModel(const std::string model_path, const std::string image_path) {
  // 1. Create CxxConfig
  auto start_time = GetCurrentUS();
  paddle::lite_api::CxxConfig cxx_config;
  cxx_config.set_model_file(model_path + "_opt/model");
  cxx_config.set_param_file(model_path + "_opt/params");
  cxx_config.set_valid_places({paddle::lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                               paddle::lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  // 2. Create PaddlePredictor by MobileConfig
  try {
    predictor = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::CxxConfig>(cxx_config);
    std::cout << "==============CxxConfig Predictor Version: " << predictor->GetVersion() << " ==============" << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }
  auto end_time = GetCurrentUS();
  // 3. Run model
  process(predictor, image_path);
  std::cout << "CXXConfig preprosss: " 
            << (end_time - start_time) / 1000.0
            << " ms." << std::endl;
}

void SaveModel(const std::string model_path, const int model_type) {
  // 1. Create CxxConfig
  paddle::lite_api::CxxConfig cxx_config;
  if (model_type) { // combined model
    cxx_config.set_model_file(model_path + "/__model__");
    cxx_config.set_param_file(model_path + "/__params__");
  } else {
    cxx_config.set_model_dir(model_path);
  }
  cxx_config.set_valid_places({paddle::lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                               paddle::lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  // cxx_config.set_subgraph_model_cache_dir(model_path.substr(0, model_path.find_last_of("/")));

  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  try {
    predictor = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::CxxConfig>(cxx_config);
    std::cout << "============== CxxConfig Predictor Version: " << predictor->GetVersion() << " ==============" << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }

  // 3. Save optimized model
  predictor->SaveOptimizedModel(model_path, paddle::lite_api::LiteModelType::kNaiveBuffer);
  std::cout << "Save optimized model to " << (model_path+".nb") << std::endl;

  predictor->SaveOptimizedModel(model_path+"_opt", paddle::lite_api::LiteModelType::kProtobuf);
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

  std::string model_path = assets_dir + "/models/" + model_name;
  std::string image_path = assets_dir + "/images/" + image_name;

  std::cout << "Model Path is <" << model_path << ">" << std::endl;
  std::cout << "Image Path is <" << image_path << ">" << std::endl;

#ifdef USE_FULL_API
  SaveModel(model_path, model_type);
  RunFullModel(model_path, image_path);
#endif

  RunLiteModel(model_path, image_path);

  return 0;
}
