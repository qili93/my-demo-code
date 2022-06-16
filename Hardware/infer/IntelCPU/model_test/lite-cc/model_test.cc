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

const std::string IMAGE_FILE_NAME = "face.jpg"; // {1, 3, 1050, 1682} NCHW
const std::string IMAGE_DATA_NAME = "face.raw"; // {1, 3, 320, 512} NCHW

// // MODEL_NAME=align150-fp32
// const std::vector<int> INPUT_SHAPE = {1, 3, 128, 128};

// // MODEL_NAME=angle-fp32
// const std::vector<int> INPUT_SHAPE = {1, 3, 64, 64};

// // MODEL_NAME=detect_rgb-fp32
const std::vector<int64_t> INPUT_SHAPE = {1, 3, 320, 512};

// // MODEL_NAME=eyes_position-fp32
// const std::vector<int> INPUT_SHAPE = {1, 3, 32, 32};

// // MODEL_NAME=iris_position-fp32
// const std::vector<int> INPUT_SHAPE = {1, 3, 24, 24};

// // MODEL_NAME=mouth_position-fp32
// const std::vector<int> INPUT_SHAPE = {1, 3, 48, 48};

// MODEL_NAME=pc-seg-float-model
// const std::vector<int> INPUT_SHAPE = {1, 4, 192, 256};

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

void preprocess(const std::string image_path, float * input_data, const int input_width, const int input_height) {
  cv::Mat input_image = cv::imread(image_path, 1);
  // resize image
  cv::Mat resize_image;
  cv::resize(input_image, resize_image, cv::Size(input_width, input_height), 0, 0); // width, height
  if (resize_image.channels() == 4) {
    cv::cvtColor(resize_image, resize_image, cv::COLOR_BGRA2RGB);
  }
  // minus mean
  cv::subtract(resize_image, cv::Scalar(104., 117., 123.), resize_image);
  // norm to 2/255
  cv::Mat norm_image;
  resize_image.convertTo(norm_image, CV_32FC3, 2 / 255.f);
  // HWC (320, 512, 3) -> CHW (3, 320, 512)
  cv::Size newsize(input_width, input_height*3); // width, height*3
  cv::Mat destination(newsize, CV_32FC1);
  for (int i = 0; i < norm_image.channels(); ++i) {
    cv::extractChannel(norm_image, cv::Mat(input_height, input_width, CV_32FC1, 
                       &(destination.at<float>(input_height * input_width * i))),i);
  }
  // std::cout << "Destination Image Size is: (" << destination.size().height << "," 
  //           <<  destination.size().width << "," << destination.channels() << ")" << std::endl;

  const float * image_data = reinterpret_cast<const float *>(destination.data);
  std::copy(image_data, image_data + ShapeProduction(INPUT_SHAPE), input_data);
}

void draw_image(cv::Mat& input_image, const float* face) {
  int image_height = 1050;
  int image_width = 1682;
  int xmin = static_cast<int>(face[2] * image_width);
  int ymin = static_cast<int>(face[3] * image_height);
  int xmax = static_cast<int>(face[4] * image_width);
  int ymax = static_cast<int>(face[5] * image_height);
  cv::Point ptmin(xmin, ymin);
  cv::Point ptmax(xmax, ymax);
  cv::rectangle(input_image, ptmin, ptmax, cv::Scalar(0, 255, 0), 2);
}

void postprocess(const float * output_data, const std::vector<int64_t>& shape) {
  int output_num = static_cast<int>(shape[0]);
  cv::Mat input_image = cv::imread("../assets/images/face.jpg", 1);
  for (int i = 0; i < output_num; ++i) {
    int offset = i * 6;
    float score = output_data[offset + 1];
    if (score < 0.5) continue;
    std::cout << data_to_string<float>(&output_data[offset], 6) << std::endl;
    // detect_output.push_back(&output_data[offset]);
    draw_image(input_image, &output_data[offset]);
  }
  cv::imwrite("face_detect.jpg", input_image);
}

void process(std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor, const std::string image_path) {
  // 1. Prepare input data
  std::unique_ptr<paddle::lite_api::Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize(INPUT_SHAPE);
  auto* input_data = input_tensor->mutable_data<float>();
  // read_imgnp(imgnp_path, input_data);
  preprocess(image_path, input_data, INPUT_SHAPE[3], INPUT_SHAPE[2]);

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
  // 5. Get all output
  int output_num = static_cast<int>(predictor->GetOutputNames().size());
  for (int i = 0; i < output_num; ++i) {
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(std::move(predictor->GetOutput(i)));
    const float *output_data = output_tensor->data<float>();
    std::cout << "Printing Output Index: <" << i << ">, shape is " << shape_to_string(output_tensor->shape()) << std::endl;
    postprocess(output_data, output_tensor->shape());
  }
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
