#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <iterator>
#include <math.h>
#include <float.h>
#include <paddle_api.h>

#if !defined(_WIN32)
#include <sys/time.h>
#else
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#include <windows.h>
#include<sys/timeb.h>
#endif

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
const int CPU_THREAD_NUM = 1;

const std::string model_align = "../assets/models/align150-fp32"; // {1, 3, 128, 128}
const std::string model_eyes = "../assets/models/eyes_position-fp32"; // {1, 3, 32, 32}
const std::string model_iris = "../assets/models/iris_position-fp32"; // {1, 3, 24, 24}
const std::string model_mouth = "../assets/models/mouth_position-fp32"; // {1, 3, 48, 48}

// MODEL_NAME=align150-fp32
const std::vector<int64_t> INPUT_SHAPE_ALIGN = {1, 3, 128, 128};
const std::vector<int64_t> INPUT_SHAPE_EYES = {1, 3, 32, 32};
const std::vector<int64_t> INPUT_SHAPE_IRIS = {1, 3, 24, 24};
const std::vector<int64_t> INPUT_SHAPE_MOUTH = {1, 3, 48, 48};

static double total_time = 0;

static std::string read_file(std::string filename) {
  FILE *file = fopen(filename.c_str(), "rb");
  if (file == nullptr) {
    std::cout << "Failed to open file: " << filename << std::endl;
    return nullptr;
  }
  fseek(file, 0, SEEK_END);
  int64_t size = ftell(file);
  if (size == 0) {
    std::cout << "File should not be empty: " << size << std::endl;
    return nullptr;
  }
  rewind(file);
  char * data = new char[size];
  size_t bytes_read = fread(data, 1, size, file);
  if (bytes_read != size) {
    std::cout << "Read binary file bytes do not match with fseek: " << bytes_read << std::endl;
    return nullptr;
  }
  fclose(file);
  std::string file_data(data, size);
  return file_data;
}

int64_t shape_production(const std::vector<int64_t>& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

#if !defined(_WIN32)
double get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}
#else
double get_current_us() {
  struct timeb cur_time;
  ftime(&cur_time);
  return (cur_time.time * 1e+6) + cur_time.millitm * 1e+3;
}
#endif

template <typename T>
std::string data_to_string(const T* data, const int64_t size) {
  std::ostringstream ss;
  ss << "[";
  for (int64_t i = 0; i < size - 1; ++i) {
    ss << std::setprecision(3) << std::setw(10) << std::setfill(' ') 
       << std::fixed << data[i] << ", ";
  }
  ss << std::setprecision(3) << std::setw(10) << std::setfill(' ') 
     << std::fixed << data[size - 1] << "]";
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

void speed_report(const std::vector<double>& costs) {
  double max = 0, min = FLT_MAX, sum = 0, avg;
  for (auto v : costs) {
      max = fmax(max, v);
      min = fmin(min, v);
      sum += v;
  }
  avg = costs.size() > 0 ? sum / costs.size() : 0;
  std::cout << "================== Speed Report ==================" << std::endl;
  std::cout << "[ - ]  warmup: " << FLAGS_warmup 
            << ", repeats: " << FLAGS_repeats 
            << ", max=" << max << " ms, min=" << min
            << "ms, avg=" << avg << "ms" << std::endl;
  // get total avg time
  total_time += avg;
}

void process(std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor, const std::vector<int64_t> INPUT_SHAPE) {
  // 1. Prepare input data
  std::unique_ptr<paddle::lite_api::Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize(INPUT_SHAPE);
  auto* input_data = input_tensor->mutable_data<float>();
  for (size_t i = 0; i < shape_production(INPUT_SHAPE); ++i) {
    input_data[i] = 1.0f;
  }

  // 2. Warmup Run
  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }
  // 3. Repeat Run
  std::vector<double> costs;
  for (int i = 0; i < FLAGS_repeats; ++i) {
    auto start_time = get_current_us();
    predictor->Run();
    auto end_time = get_current_us();
    costs.push_back((end_time - start_time) / 1000.0);
  }

  // 5. Get all output
  int output_num = static_cast<int>(predictor->GetOutputNames().size());
  for (int i = 0; i < output_num; ++i) {
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(std::move(predictor->GetOutput(i)));
    const float *output_data = output_tensor->data<float>();
    const int64_t ouput_size = shape_production(output_tensor->shape());
    std::cout << "Output Index: <" << i << ">, shape: " << shape_to_string(output_tensor->shape()) << std::endl;
    for (int j = 0; j < ouput_size; ++j) {
      std::cout << std::setprecision(3) << std::setw(10) << std::setfill(' ')  << std::fixed << output_data[i] << std::endl;
    }
  }

  // 5. speed report
  speed_report(costs);
}

void RunLiteModel(const std::string model_path, const std::vector<int64_t> INPUT_SHAPE) {
  // 1. Create MobileConfig
  auto start_time = get_current_us();
  paddle::lite_api::MobileConfig mobile_config;
  // mobile_config.set_model_from_file(model_path+".nb");
  // Load model from buffer
  std::string model_buffer = read_file(model_path+".nb");
  std::cout << "model_buffer length is " << model_buffer.length() << std::endl;
  mobile_config.set_model_from_buffer(model_buffer);
  mobile_config.set_threads(CPU_THREAD_NUM);
  mobile_config.set_power_mode(paddle::lite_api::PowerMode::LITE_POWER_HIGH);
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  try {
    predictor = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(mobile_config);
    // std::cout << "MobileConfig Predictor Version: " << predictor->GetVersion() << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(mobile config)." << std::endl;
  }
  auto end_time = get_current_us();
  // 3. Run model
  process(predictor, INPUT_SHAPE);
  // std::cout << "MobileConfig preprosss: " << (end_time - start_time) / 1000.0 << " ms." << std::endl;
}

#ifdef USE_FULL_API
void SaveOptModel(const std::string model_path, const int model_type = 0) {
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
    std::cout << "CxxConfig Predictor Version: " << predictor->GetVersion() << std::endl;
  } catch (std::exception e) {
    std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
  }

  // 3. Save optimized model
  predictor->SaveOptimizedModel(model_path, paddle::lite_api::LiteModelType::kNaiveBuffer);
  std::cout << "Save optimized model to " << (model_path+".nb") << std::endl;

  // predictor->SaveOptimizedModel(model_path+"_opt", paddle::lite_api::LiteModelType::kProtobuf);
  // std::cout << "Save optimized model to " << (model_path+"_opt") << std::endl;
}
#endif

int main(int argc, char **argv) {

  std::cout << "Model Path is <" << model_align << ">" << std::endl;
  std::cout << "Input Shape is " << shape_to_string(INPUT_SHAPE_ALIGN) << std::endl;
#ifdef USE_FULL_API
  SaveOptModel(model_align, 1);
#endif
  RunLiteModel(model_align, INPUT_SHAPE_ALIGN);

  std::cout << std::endl;
  std::cout << "Model Path is <" << model_eyes << ">" << std::endl;
  std::cout << "Input Shape is " << shape_to_string(INPUT_SHAPE_EYES) << std::endl;
#ifdef USE_FULL_API
  SaveOptModel(model_eyes, 1);
#endif
  RunLiteModel(model_eyes, INPUT_SHAPE_EYES);

  std::cout << std::endl;
  std::cout << "Model Path is <" << model_iris << ">" << std::endl;
  std::cout << "Input Shape is " << shape_to_string(INPUT_SHAPE_IRIS) << std::endl;
#ifdef USE_FULL_API
  SaveOptModel(model_iris, 1);
#endif
  RunLiteModel(model_iris, INPUT_SHAPE_IRIS);

  std::cout << std::endl;
  std::cout << "Model Path is <" << model_mouth << ">" << std::endl;
  std::cout << "Input Shape is " << shape_to_string(INPUT_SHAPE_MOUTH) << std::endl;
#ifdef USE_FULL_API
  SaveOptModel(model_mouth, 1);
#endif
  RunLiteModel(model_mouth, INPUT_SHAPE_MOUTH);

  std::cout << "==========================================" << std::endl;
  std::cout << "Total AVG TIME is " << total_time <<  " ms" << std::endl;

  return 0;
}
