#include <cstdlib>
#include <ctime>
#include "device.h"
#include "compute.h"
#include "utils.h"

 bool DeviceProgram::LoadFromCacheFile(const std::string& model_cache_dir) {
  //auto device_ = std::make_shared<Device>();
  //model_client_ = my_lite_demo::Device::Global().LoadFromFile(model_cache_dir);
  LOG(INFO) << "[ASCEND] Staring LoadFromCacheFile ...";
  //my_lite_demo::Device my_device;
  model_client_ = my_lite_demo::Device::Global().LoadFromFile(model_cache_dir);
  if (!model_client_) {
    LOG(WARNING) << "[ASCEND] Load model from cached file failed!";
    return false;
  }
  LOG(INFO) << "[ASCEND] Load model from cached file success.";
  LOG(INFO) << "[ASCEND] Finishing LoadFromCacheFile ...";
  return true;
 }

bool DeviceProgram::BuildGraphAndCacheToFile(ge::Graph &om_graph, const std::string& model_cache_dir){
    LOG(INFO) << "[ASCEND] Staring BuildGraphAndCacheToFile ...";
    // 2. system init
    std::map<std::string, std::string> global_options = {
        {ge::ir_option::SOC_VERSION, "Ascend310"},
    };
    if (ge::aclgrphBuildInitialize(global_options) !=  ge::GRAPH_SUCCESS) {
      LOG(ERROR) << "[ASCEND] aclgrphBuildInitialize Failed!";
    }
    LOG(INFO) << "[ASCEND] aclgrphBuildInitialize success!";

    // 3. Build IR Model
    ge::ModelBufferData model_om_buffer;
    std::map<std::string, std::string> options;
    //PrepareOptions(options);

    if (ge::aclgrphBuildModel(om_graph, options, model_om_buffer) !=  ge::GRAPH_SUCCESS) {
      LOG(ERROR) << "[ASCEND] aclgrphBuildModel Failed!";
    } else {
      LOG(INFO) << "[ASCEND] aclgrphBuildModel success!";
    }

    // 4. Save IR Model
    if (ge::aclgrphSaveModel(model_cache_dir, model_om_buffer) != ge::GRAPH_SUCCESS) {
      LOG(ERROR) << "[ASCEND] aclgrphSaveModel Failed!";
    } else {
      LOG(INFO) << "[ASCEND] success saving model to " << model_cache_dir;
    }

    // 5. release resource
    ge::aclgrphBuildFinalize();
    
    //my_lite_demo::Device my_device;
    std::vector<char> model_buffer(model_om_buffer.data.get(), model_om_buffer.data.get()+model_om_buffer.length);
    model_client_ = my_lite_demo::Device::Global().LoadFromMem(model_buffer);
    if (!model_client_) {
      LOG(WARNING) << "[ASCEND] Load model from memory failed!";
      return false;
    }
    LOG(INFO) << "[ASCEND] create model_client and load model from memory success.";
    LOG(INFO) << "[ASCEND] Finishing BuildGraphAndCacheToFile ...";
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
  LOG(INFO) << "[ASCEND] resize device_itensors to " << device_idims_.size();
  for (size_t i = 0; i < device_idims_.size(); i++) {
    LOG(INFO) << "[ASCEND] Inputs[" << i << "] device dims: {" 
            << device_idims_[i].GetNumber() << ","
            << device_idims_[i].GetChannel() << ","
            << device_idims_[i].GetHeight() << ","
            << device_idims_[i].GetWidth() << "}";
    device_itensors[i].reset(new ge::Tensor(device_idims_[i].GetGeTensorDesc()));

    int64_t data_shape = device_idims_[i].GetGeTensorDesc().GetShape().GetShapeSize();
    int64_t data_length = data_shape * sizeof(float);
    LOG(INFO) << "[ASCEND] Set Input Tensor Shape is: " << data_shape;
    LOG(INFO) << "[ASCEND] Set Input Tensor Size is: " << data_length;

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
  LOG(INFO) << "[ASCEND] resize device_otensors to " << device_odims_.size();
  for (size_t i = 0; i < device_odims_.size(); i++) {
    LOG(INFO) << "[ASCEND] Output[" << i << "] device dims: {" 
            << device_odims_[i].GetNumber() << ","
            << device_odims_[i].GetChannel() << ","
            << device_odims_[i].GetHeight() << ","
            << device_odims_[i].GetWidth() << "}";
    device_otensors[i].reset(new ge::Tensor(device_odims_[i].GetGeTensorDesc()));

    int64_t data_shape = device_odims_[i].GetGeTensorDesc().GetShape().GetShapeSize();
    int64_t data_length = data_shape * sizeof(float);
    LOG(INFO) << "[ASCEND] Set Output Tensor Shape is: " << data_shape;
    LOG(INFO) << "[ASCEND] Set Output Tensor Size is: " << data_length;
    // float data_value = 0.5;
    // auto status = device_itensors[i]->SetData(reinterpret_cast<uint8_t*>(&data_value), data_length);
    // if (status != ge::GRAPH_SUCCESS) {
    //   LOG(INFO) << "Set Input Tensor Data Failed";
    //   return false;
    // }
  }

  // ge::TensorDesc output_desc(ge::Shape({ 1, 1, 3, 3 }), ge::FORMAT_ND, ge::DT_FLOAT);
  // ge::Tensor output_y(output_desc);
  // device_otensors.push_back(std::make_shared<ge::Tensor>(output_y));
  LOG(INFO) << "[ASCEND] Finishing InitDeivceTensors ...";
  return true;
}

bool DeviceProgram::ZeroCopyRun(std::vector<std::shared_ptr<ge::Tensor>>* device_itensors,
                 std::vector<std::shared_ptr<ge::Tensor>>* device_otensors) {
  LOG(INFO) << "[ASCEND] Staring ZeroCopyRun ...";
  model_client_->ModelExecute(device_itensors, device_otensors);
  my_lite_demo::Device::Global().ReleaseDevice();
  LOG(INFO) << "[ASCEND] Finishing ZeroCopyRun ...";
  return true;
}