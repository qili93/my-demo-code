#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iomanip>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <paddle_api.h>

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
const int CPU_THREAD_NUM = 1;

// const std::string model_path = "../assets/models/align150-fp32"; // {1, 3, 128, 128}
// const std::string model_path = "../assets/models/align150-fp32-dst"; //  {1, 8, 64, 64}
// const std::string model_path = "../assets/models/inference_model"; // {1, 3, 128, 128}
const std::string model_path = "../assets/models/dconv08"; // {1, 3, 128, 128}

// MODEL_NAME=align150-fp32
const std::vector<int64_t> INPUT_SHAPE = {1, 8, 64, 64};
// const std::vector<int64_t> INPUT_SHAPE = {1, 8, 4, 4};

// static double total_time = 0; 

int64_t shape_production(const std::vector<int64_t>& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

double get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

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
  // ss << data[size - 1] << "]";
  return ss.str();
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
  std::ostringstream ss;
  if (shape.empty()) {
    ss << "{}";
    return ss.str();
  }
  ss << "{";
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    ss << shape[i] << ", ";
  }
  ss << shape[shape.size() - 1] << "}";
  return ss.str();
}

// template <typename T>
// std::string tensor_to_string(const T* data, const std::vector<int64_t>& shape) {
//   std::ostringstream ss;
//   ss << "Shape: " << shape_to_string(shape) << std::endl;
//   const int64_t stride = shape.back();
//   const int64_t split = shape.size() > 2 ? shape[shape.size() - 2] : 0;
//   const int64_t length = static_cast<int64_t>(shape_production(shape) / stride);
//   for (size_t i = 0; i < length; ++i) {
//     const T * data_start = data + i * stride;
//     ss << data_to_string<T>(data_start, stride) << std::endl;
//     if (split != 0 && ((i + 1) % split) == 0) {
//       ss <<  std::endl;
//     }
//   }
//   return ss.str();
// }

template <typename T>
std::string tensor_to_string(const T* data, const int64_t size) {
  std::ostringstream ss;
  for (size_t i = 0; i < size; ++i) {
    ss << std::setprecision(3) << std::setw(10) << std::setfill(' ') 
       << std::fixed << data[i] << std::endl;
  }
  return ss.str();
}

void speed_report(const std::vector<float>& costs) {
  float max = 0, min = FLT_MAX, sum = 0, avg;
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
}

void read_rawfile(const std::string raw_imgnp_path, float * input_data) {
  std::ifstream raw_imgnp_file(raw_imgnp_path, std::ios::in | std::ios::binary);
  if (!raw_imgnp_file) {
    std::cout << "Failed to load raw image file: " <<  raw_imgnp_path << std::endl;
    return;
  }
  int64_t raw_imgnp_size = shape_production(INPUT_SHAPE);
  raw_imgnp_file.read(reinterpret_cast<char *>(input_data), raw_imgnp_size * sizeof(float));
  raw_imgnp_file.close();
}

// save output to raw file
void write_rawfile(const float * output_data, const int64_t output_size, const std::string rawfile) {
  std::ofstream output_file(rawfile, std::ios::out | std::ios::trunc );
  if (!output_file.is_open()) {
    std::cout << "Failed to open raw output file: " << rawfile << std::endl;
    return;
  }
  output_file.write(reinterpret_cast<const char *>(output_data), output_size * sizeof(float));
  output_file.close();
}

void compare_rawfile(const float * output_data, const int64_t output_size, const std::string rawfile) {
  float* infer_out = new float[output_size];
  std::ifstream infer_out_file(rawfile, std::ios::in | std::ios::binary);
  if (!infer_out_file.is_open()) {
    std::cout << "Failed to open raw input file: " << rawfile << std::endl;
    return;
  }
  infer_out_file.read(reinterpret_cast<char *>(infer_out), output_size * sizeof(float));
  for (int i = 0; i < output_size; ++i) {
    if (fabs(output_data[i] - infer_out[i]) > 0.002) {
      std::cout << "abs error exceeded: index " << i << ", infer res is " << infer_out[i] << ", lite res is " << output_data[i] << std::endl;
    }
  }
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
  std::vector<float> costs;
  for (int i = 0; i < FLAGS_repeats; ++i) {
    auto start_time = get_current_us();
    predictor->Run();
    auto end_time = get_current_us();
    costs.push_back((end_time - start_time) / 1000.0);
  }

  // 4. Get all output
  std::cout << "Input Index: <0>, shape: " << shape_to_string(INPUT_SHAPE) << std::endl;
  // std::cout << tensor_to_string(input_data, INPUT_SHAPE) << std::endl;
  int output_num = static_cast<int>(predictor->GetOutputNames().size());
  for (int i = 0; i < output_num; ++i) {
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(std::move(predictor->GetOutput(i)));
    const float *output_data = output_tensor->data<float>();
    const int64_t ouput_size = shape_production(output_tensor->shape());
    std::cout << "Output Index: <" << i << ">, shape: " << shape_to_string(output_tensor->shape()) << std::endl;
    // std::cout << tensor_to_string(output_data, output_tensor->shape()) << std::endl;
    std::ofstream lite_out("lite-out-"+std::to_string(i)+".txt");
    lite_out << tensor_to_string(output_data, ouput_size);
    lite_out.close();
  }

  // 5. speed report
  speed_report(costs);
}

void RunLiteModel(const std::string model_path, const std::vector<int64_t> INPUT_SHAPE) {
  // 1. Create MobileConfig
  auto start_time = get_current_us();
  paddle::lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(model_path+".nb");
  // Load model from buffer
  // std::string model_buffer = ReadFile(model_path+".nb");
  // mobile_config.set_model_from_buffer(model_buffer);
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

  predictor->SaveOptimizedModel(model_path+"_opt", paddle::lite_api::LiteModelType::kProtobuf);
  std::cout << "Save optimized model to " << (model_path+"_opt") << std::endl;
}
#endif

int main(int argc, char **argv) {

  std::cout << "Model Path is <" << model_path << ">" << std::endl;
  std::cout << "Input Shape is " << shape_to_string(INPUT_SHAPE) << std::endl;
#ifdef USE_FULL_API
  SaveOptModel(model_path, 1);
#endif
  RunLiteModel(model_path, INPUT_SHAPE);

  return 0;
}
