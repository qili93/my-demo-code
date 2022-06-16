#include <gflags/gflags.h>
#include <glog/logging.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <memory>
#include <sys/time.h>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT
#include <condition_variable>  // NOLINT
#include <paddle_inference_api.h>

//mul_model
const std::vector<int> INPUT_SHAPE = {1, 1};
const std::string FLAGS_infer_model = "../assets/models/mul_model";

class Barrier {
 public:
  explicit Barrier(std::size_t count) : _count(count) {}
  void Wait() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (--_count) {
      _cv.wait(lock, [this] { return _count == 0; });
    } else {
      _cv.notify_all();
    }
  }

 private:
  std::mutex _mutex;
  std::condition_variable _cv;
  std::size_t _count;
};

int shape_production(const std::vector<int>& shape) {
  int res = 1;
  for (auto i : shape) res *= i;
  return res;
}

static std::string shape_to_string(const std::vector<int>& shape) {
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

int test_main(const paddle_infer::Config& config, Barrier* barrier = nullptr) {
  static std::mutex mutex;
  std::shared_ptr<paddle_infer::Predictor> predictor;
  {
    std::unique_lock<std::mutex> lock(mutex);
    predictor = std::move(paddle_infer::CreatePredictor(config));
  }
  if (barrier) {
    barrier->Wait();
  }

  // 3. Prepare input data
  int input_num = shape_production(INPUT_SHAPE);
  std::vector<float> input_data(input_num, 1);

  auto input_names = predictor->GetInputNames();
  auto input_tensor = predictor->GetInputHandle(input_names[0]);
  input_tensor->Reshape(INPUT_SHAPE);
  input_tensor->CopyFromCpu(input_data.data());

  predictor->Run();

  auto output_names = predictor->GetOutputNames();
  auto output_tensor = predictor->GetInputHandle(output_names[0]);
  std::vector<int> output_shape = output_tensor->shape();
  std::cout << "Output shape is " << shape_to_string(output_shape) << std::endl;

  // int output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  // std::vector<float> output_data;
  // output_data.resize(output_size);
  // output_tensor->CopyToCpu(output_data.data());
  // std::cout << "Output data size is " << output_data.size() << std::endl;
}

int main(int argc, char **argv) {
  const size_t thread_num = 5;
  std::vector<std::thread> threads(thread_num);
  Barrier barrier(thread_num);
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&barrier, i]() {
      paddle_infer::Config config;
      config.EnableUseGpu(100, 0);
      config.SetModel(FLAGS_infer_model);
      config.EnableGpuMultiStream();
      test_main(config, &barrier); // main function to create predictor and run
    });
  }
  for (auto& th : threads) {
    th.join();
  }
}
