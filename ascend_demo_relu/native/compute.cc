#include "device.h"
#include "compute.h"
#include "utils.h"

 bool DeviceProgram::LoadFromCacheFile(const std::string& model_cache_dir) {
  //auto device_ = std::make_shared<Device>();
  //model_client_ = my_lite_demo::Device::Global().LoadFromFile(model_cache_dir);
  INFO_LOG("-------Enter: [compute](LoadFromCacheFile)-------");
  //my_lite_demo::Device my_device;
  model_client_ = my_lite_demo::Device::Global().LoadFromFile(model_cache_dir);
  if (!model_client_) {
    WARN_LOG("[compute](LoadFromCacheFile) Load model from cached file failed!");
    return false;
  }
  INFO_LOG("[compute](LoadFromCacheFile) Load model from cached file success.");
  INFO_LOG("-------Leave: [compute](LoadFromCacheFile)-------");
  return true;
 }

bool DeviceProgram::BuildGraphAndCacheToFile(ge::Graph &om_graph, const std::string& model_cache_dir){
    INFO_LOG("-------Enter: [compute](BuildGraphAndCacheToFile)-------");
    // 2. system init
    std::map<std::string, std::string> global_options = {
        {ge::ir_option::SOC_VERSION, "Ascend310"},
    };
    if (ge::aclgrphBuildInitialize(global_options) !=  ge::GRAPH_SUCCESS) {
      ERROR_LOG("[compute](BuildGraphAndCacheToFile) aclgrphBuildInitialize Failed!");
    }
    INFO_LOG("[compute](BuildGraphAndCacheToFile) aclgrphBuildInitialize SUCCESS!");

    // 3. Build IR Model
    ge::ModelBufferData model_om_buffer;
    std::map<std::string, std::string> options;
    //PrepareOptions(options);

    if (ge::aclgrphBuildModel(om_graph, options, model_om_buffer) !=  ge::GRAPH_SUCCESS) {
      ERROR_LOG("[compute](BuildGraphAndCacheToFile) aclgrphBuildModel Failed!");
    } else {
      INFO_LOG("[compute](BuildGraphAndCacheToFile) aclgrphBuildModel SUCCESS!");
    }

    // 4. Save IR Model
    if (ge::aclgrphSaveModel(model_cache_dir, model_om_buffer) != ge::GRAPH_SUCCESS) {
      ERROR_LOG("[compute](BuildGraphAndCacheToFile) aclgrphSaveModel Failed!");
    } else {
      INFO_LOG("[compute](BuildGraphAndCacheToFile) aclgrphSaveModel SUCCESS!");
    }

    // 5. release resource
    ge::aclgrphBuildFinalize();
    INFO_LOG("-------Leave: [compute](BuildGraphAndCacheToFile)-------");

    // Load the om model and create a model manager client
    //model_client_ = my_lite_demo::Device::Global().LoadFromMem(model_om_buffer);

    return true;
}

bool DeviceProgram::InitDeivceTensors(std::vector<std::shared_ptr<ge::Tensor>>& device_itensors,
                 std::vector<std::shared_ptr<ge::Tensor>>& device_otensors) {
  ge::TensorDesc input_desc(ge::Shape({ 1, 1, 2, 2 }), ge::FORMAT_ND, ge::DT_FLOAT);
  float *input_data = new float[4];
  for (int i = 0; i < 4; ++i) {
    input_data[i] = static_cast<float>(i) - 1.5;
  }
  ge::Tensor input_x(input_desc, reinterpret_cast<uint8_t*>(input_data), 4*sizeof(float));
  device_itensors.push_back(std::make_shared<ge::Tensor>(input_x));

  ge::TensorDesc output_desc(ge::Shape({ 1, 1, 2, 2 }), ge::FORMAT_ND, ge::DT_FLOAT);
  float *output_data = new float[4];
  for (int i = 0; i < 4; ++i) {
    *(input_data+i) = 1;
  }
  ge::Tensor output_y(output_desc, reinterpret_cast<uint8_t*>(output_data), 4*sizeof(float));
  device_otensors.push_back(std::make_shared<ge::Tensor>(output_y));
  return true;
}

bool DeviceProgram::ZeroCopyRun(std::vector<std::shared_ptr<ge::Tensor>>* device_itensors,
                 std::vector<std::shared_ptr<ge::Tensor>>* device_otensors) {
  model_client_->ModelExecute(*device_itensors, *device_otensors);
  my_lite_demo::Device::Global().ReleaseDevice();
  return true;
}