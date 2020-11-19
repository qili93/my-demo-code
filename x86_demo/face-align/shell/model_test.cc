#include <iostream>
#include <sstream>
#include <algorithm>
#include <sys/time.h>
#include <paddle_api.h>

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
  auto start_time = get_current_us();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor->Run();
  }
  auto end_time = get_current_us();

  auto avg_time = (end_time - start_time) / FLAGS_repeats / 1000.0;
  total_time += avg_time;

  // 4. Speed Report
  std::cout << "------- Speed Report -------" << std::endl;
  std::cout << "Warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << (end_time - start_time) / FLAGS_repeats / 1000.0
            << " ms in average." << std::endl;

  // 5. Get all output
  int output_num = static_cast<int>(predictor->GetOutputNames().size());
  for (int i = 0; i < output_num; ++i) {
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(std::move(predictor->GetOutput(i)));
    const float *output_data = output_tensor->data<float>();
    const int64_t ouput_size = shape_production(output_tensor->shape());
    std::cout << "Output Index: <" << i << ">, shape: " << shape_to_string(output_tensor->shape()) << std::endl;
    // LOG(INFO) << data_to_string<float>(output_data, ShapeProduction(output_tensor->shape())) << std::endl;
  }
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

int main(int argc, char **argv) {

  std::cout << "Model Path is <" << model_align << ">" << std::endl;
  std::cout << "Input Shape is " << shape_to_string(INPUT_SHAPE_ALIGN) << std::endl;
  RunLiteModel(model_align, INPUT_SHAPE_ALIGN);

  std::cout << std::endl;
  std::cout << "Model Path is <" << model_eyes << ">" << std::endl;
  std::cout << "Input Shape is " << shape_to_string(INPUT_SHAPE_EYES) << std::endl;
  RunLiteModel(model_eyes, INPUT_SHAPE_EYES);

  std::cout << std::endl;
  std::cout << "Model Path is <" << model_iris << ">" << std::endl;
  std::cout << "Input Shape is " << shape_to_string(INPUT_SHAPE_IRIS) << std::endl;
  RunLiteModel(model_iris, INPUT_SHAPE_IRIS);

  std::cout << std::endl;
  std::cout << "Model Path is <" << model_mouth << ">" << std::endl;
  std::cout << "Input Shape is " << shape_to_string(INPUT_SHAPE_MOUTH) << std::endl;
  RunLiteModel(model_mouth, INPUT_SHAPE_MOUTH);

  std::cout << "==========================================" << std::endl;
  std::cout << "Total AVG TIME is " << total_time <<  " ms" << std::endl;

  return 0;
}
