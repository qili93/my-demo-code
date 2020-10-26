#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "paddle_api.h"
#include "logging.h"

using namespace paddle::lite_api;  // NOLINT

const int FLAGS_warmup = 5;
const int FLAGS_repeats = 10;
const int CPU_THREAD_NUM = 1;

void preprocess(const std::string image_path) {
  // float data[24];
  // for (int i = 0; i < 8; i ++) {
  //   data[i * 3] = i;
  //   data[i * 3 + 1] = i;
  //   data[i * 3 + 2] = i;
  // }
  cv::Mat input_image = cv::imread(image_path, 1);
  // cv::Mat input_image(4, 2, CV_32FC3, data); // height, width, channel=3
  LOG(INFO) << "input_image.channels()=" << input_image.channels();
  LOG(INFO) << "input_image.size().height=" << input_image.size().height;
  LOG(INFO) << "input_image.size().width=" << input_image.size().width;
  // std::cout << "input_image = " << std::endl <<  input_image << std::endl;

  // resize height to 320
  cv::Mat resize_image;
  float scale = 320.0 / input_image.size().height;
  int resize_height = static_cast<int>(input_image.size().height * scale);
  int resize_width = static_cast<int>(input_image.size().width * scale);
  cv::resize(input_image, resize_image, cv::Size(resize_width, resize_height), 0, 0); // width, height

  LOG(INFO) << "resize_image.channels()=" << resize_image.channels();
  LOG(INFO) << "resize_image.size().height=" << resize_image.size().height;
  LOG(INFO) << "resize_image.size().width=" << resize_image.size().width;

  // cv::subtract(input_image, cv::Scalar(1., 2., 3.), input_image);
  // std::cout << "input_image = " << std::endl <<  input_image << std::endl;
  cv::subtract(resize_image, cv::Scalar(104., 117., 123.), resize_image);
  cv::multiply(resize_image, cv::Scalar( 0.007843), resize_image);

  //copy the channels from the source image to the destination # HWC to CHW
  // cv::Size input_size = input_image.size();
  // cv::Size newsize(input_size.width,input_size.height*3);
  cv::Size newsize(resize_width, resize_height*3);
  cv::Mat destination(newsize, CV_32FC1);
  for (int i = 0; i < resize_image.channels(); ++i) {
    cv::extractChannel(resize_image, cv::Mat(resize_height, resize_width, CV_32FC1, 
                       &(destination.at<float>(resize_height * resize_width * i))),i);
  }
  LOG(INFO) << "destination.channels()=" << destination.channels();
  LOG(INFO) << "destination.size().height=" << destination.size().height;
  LOG(INFO) << "destination.size().width=" << destination.size().width;

  // std::cout << "destination = " << std::endl <<  destination << std::endl;
  // const std::vector<float> INPUT_MEAN = {104., 117., 123.};
  // const float INPUT_SCALE = 0.007843;

  
  // cv::mul(destination, 3);
  // cv::multiply(destination, cv::Scalar(0.1), destination);

  // std::cout << "destination = " << std::endl <<  destination << std::endl;

  const float *image_data = reinterpret_cast<const float *>(destination.data);

  // for (int i = 0; i < 24; ++i) {
  //   std::cout << "image_data[" << i << "] = " << image_data[i] << std::endl;
  // }

  // for (int i = 0; i < destination.channels(); ++i) {

  // }
  

  // cv::Mat resize_image;
  // cv::resize(input_image, resize_image, cv::Size(input_width, input_height), 0, 0);
  // if (resize_image.channels() == 4) {
  //   cv::cvtColor(resize_image, resize_image, cv::COLOR_BGRA2RGB);
  // }
  // cv::Mat norm_image;
  // resize_image.convertTo(norm_image, CV_32FC3, 1 / 255.f);
  // // NHWC->NCHW

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

void process(std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor, const std::vector<int64_t> input_shape_vec) {
  // 1. Prepare input data
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize(input_shape_vec);
  auto* input_data = input_tensor->mutable_data<float>();
  for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
    input_data[i] = 1.0 + i;
  }
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

void RunLiteModel(const std::string model_path, const std::vector<int64_t> input_shape_vec) {
  std::cout << "Entering RunLiteModel ..." << std::endl;
  // 1. Create MobileConfig
  auto start_time = GetCurrentUS();
  MobileConfig mobile_config;
  // mobile_config.set_model_from_file(model_path+".nb");
  // Load model from buffer
  std::string model_buffer = ReadFile(model_path+".nb");
  mobile_config.set_model_from_buffer(model_buffer);
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
  process(predictor, input_shape_vec);
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

  // set input shape based on model name
  std::vector<int64_t> input_shape_vec(4);
  if (model_name == "align150-fp32") {
    int64_t input_shape[] = {1, 3, 128, 128};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "angle-fp32") {
    int64_t input_shape[] = {1, 3, 64, 64};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "detect_rgb-fp32") {
    int64_t input_shape[] = {1, 3, 320, 240};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "detect_rgb-int8") {
    int64_t input_shape[] = {1, 3, 320, 240};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "eyes_position-fp32") {
    int64_t input_shape[] = {1, 3, 32, 32};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "iris_position-fp32") {
    int64_t input_shape[] = {1, 3, 24, 24};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "mouth_position-fp32") {
    int64_t input_shape[] = {1, 3, 48, 48};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "seg-model-int8") {
    int64_t input_shape[] = {1, 4, 192, 192};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "pc-seg-float-model") {
    int64_t input_shape[] = {1, 4, 192, 256};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else if (model_name == "softmax_infer") {
    int64_t input_shape[] = {1, 2, 3, 1};
    std::copy (input_shape, input_shape+4, input_shape_vec.begin());
  } else {
    LOG(ERROR) << "NOT supported model name!";
    return 0;
  }

  LOG(INFO) << "Model Name is <" << model_name << ">, Input Shape is {" 
    << input_shape_vec[0] << ", " << input_shape_vec[1] << ", " 
    << input_shape_vec[2] << ", " << input_shape_vec[3] << "}";

  std::string model_path = assets_dir + "/models/" + model_name;
  std::string image_path = assets_dir + "/images/" + image_name;

  LOG(INFO) << "Model Path is <" << model_path << ">";

// #ifdef USE_FULL_API
//   SaveModel(model_path, model_type, input_shape_vec);
//   RunFullModel(model_path, input_shape_vec);
// #endif

//   RunLiteModel(model_path, input_shape_vec);

  preprocess(image_path);

  return 0;
}