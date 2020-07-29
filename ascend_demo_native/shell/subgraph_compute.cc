#include <sys/time.h>
#include <time.h>
#include "subgraph_compute.h"

 bool DeviceProgram::LoadFromCacheFile(const std::string& model_cache_dir) {
  //auto device_ = std::make_shared<Device>();
  //model_client_ = my_lite_demo::Device::Global().LoadFromFile(model_cache_dir);
  LOG(INFO) << "[ASCEND] Staring LoadFromCacheFile ...";
  //my_lite_demo::Device my_device;
  model_client_ = my_lite_demo::Device::Global().LoadFromFile(model_cache_dir, device_id_);
  if (!model_client_) {
    LOG(WARNING) << "[ASCEND] Load model from cached file failed!";
    return false;
  }
  LOG(INFO) << "[ASCEND] Load model from cached file success.";
  LOG(INFO) << "[ASCEND] Finishing LoadFromCacheFile ...";
  return true;
 }

bool DeviceProgram::BuildGraphAndCacheToFile(const std::string& model_cache_dir){
  LOG(INFO) << "[ASCEND] Staring BuildGraphAndCacheToFile ...";

  // Build IR model and save to cache file
  std::vector<char> model_buffer;
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Building model from model buffer...";
  if (!my_lite_demo::Device::Global().Build(&model_buffer, model_cache_dir)) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Build model failed!";
    return false;
  }
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Build model success.";

  // Load the om model and create a model manager client
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Loading model from memory ...";
  model_client_ = my_lite_demo::Device::Global().LoadFromMem(model_buffer, device_id_);
  if (!model_client_) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Load model from memory failed!";
    return false;
  }
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Load model from memory success.";

  return true;
}

bool DeviceProgram::InitDeivceTensors(std::vector<std::shared_ptr<ge::Tensor>>& device_itensors,
                 std::vector<std::shared_ptr<ge::Tensor>>& device_otensors) {
  LOG(INFO) << "[ASCEND] Staring InitDeivceTensors ...";

  if (device_idims_.empty() || device_odims_.empty()) {
    if (!(model_client_->GetModelIOTensorDim(&device_idims_, &device_odims_))) {
      LOG(WARNING) << "[ASCEND] Get the dimensions of input and output tensors failed!";
      return false;
    }
  }
  LOG(INFO) << "[ASCEND] GetModelIOTensorDim success.";

  device_itensors.resize(device_idims_.size());
  LOG(INFO) << "[ASCEND] resize device_itensors number to " << device_idims_.size();
  for (size_t i = 0; i < device_idims_.size(); i++) {
    LOG(INFO) << "[ASCEND] Inputs[" << i << "] device dims:" << device_idims_[i].repr();
    device_itensors[i].reset(new ge::Tensor(device_idims_[i].GetGeTensorDesc()));

    int64_t data_shape = device_idims_[i].GetGeTensorDesc().GetShape().GetShapeSize();
    int64_t data_length = data_shape * sizeof(float);
    LOG(INFO) << "[ASCEND] Input Tensor Shape Size is: " << data_shape;
    LOG(INFO) << "[ASCEND] Input Tensor Data Size is: " << data_length;

    // generating random data to input tensor between -1 to 1
    srand (static_cast <unsigned> (time(0)));
    float * pdata = new(std::nothrow) float[data_shape];
    for (int64_t j = 0; j < data_shape; j++) {
      pdata[j] = -1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/2));
    }
    auto status = device_itensors[i]->SetData(reinterpret_cast<uint8_t*>(pdata), data_length);
    if (status != ge::GRAPH_SUCCESS) {
      LOG(INFO) << "Set Input Tensor Data Failed";
      delete [] pdata;
      return false;
    }
  }

  device_otensors.resize(device_odims_.size());
  LOG(INFO) << "[ASCEND] resize device_otensors number to " << device_odims_.size();
  for (size_t i = 0; i < device_odims_.size(); i++) {
    LOG(INFO) << "[ASCEND] Output[" << i << "] device dims: " << device_odims_[i].repr(); 
    device_otensors[i].reset(new ge::Tensor(device_odims_[i].GetGeTensorDesc()));

    int64_t data_shape = device_odims_[i].GetGeTensorDesc().GetShape().GetShapeSize();
    int64_t data_length = data_shape * sizeof(float);
    LOG(INFO) << "[ASCEND] Output Tensor Shape Size is: " << data_shape;
    LOG(INFO) << "[ASCEND] Output Tensor Data Size is: " << data_length;
  }

  LOG(INFO) << "[ASCEND] Finishing InitDeivceTensors ...";
  return true;
}

bool DeviceProgram::ZeroCopyRun(std::vector<std::shared_ptr<ge::Tensor>>* device_itensors,
                 std::vector<std::shared_ptr<ge::Tensor>>* device_otensors) {
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  auto start_time = GetCurrentUS();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Starting ZeroCopyRun to ModelExecute ...";
  CHECK_EQ(model_client_->ModelExecute(device_itensors, device_otensors), true);
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Process cost " << GetCurrentUS() - start_time << " us";
  // unload model after model execution
  CHECK_EQ(model_client_->UnloadModel(), true);
  LOG(INFO) << "[ASCEND] Finishing ZeroCopyRun ...";
  return true;
}