#include <string.h>
#include <iostream>
#include <fstream>

#include "device.h"

extern bool GenYoloV3Graph(ge::Graph& graph);
extern bool GenConcatGraph(ge::Graph& graph);
extern bool GenConcatDGraph(ge::Graph& graph);

namespace my_lite_demo {

std::shared_ptr<AclModelClient> Device::LoadFromMem(
    const std::vector<char>& model_buffer, const int device_id) {
  if (model_buffer.size() == 0) {
    LOG(ERROR) << "[HUAWEI_ASCEND_NPU] model_buffer size is ZERO!";
    return nullptr;
  }

  // Create a ACL model  client to load the om model
  std::shared_ptr<AclModelClient> model_client(new AclModelClient(device_id));
  // Load model from memory
  if (model_client->LoadFromMem(
          reinterpret_cast<const void*>(model_buffer.data()),
          model_buffer.size())) {
    return model_client;
  }
  return nullptr;
}

std::shared_ptr<AclModelClient> Device::LoadFromFile(
    const std::string& model_path, const int device_id) {
  std::ifstream fs(model_path);
  if (!fs.is_open()) {
    VLOG(3) << "[HUAWEI_ASCEND_NPU] om model file not exists:" << model_path;
    return nullptr;
  }

  // Create a ACL model  client to load the om model
  std::shared_ptr<AclModelClient> model_client(new AclModelClient(device_id));
  // Load model from memory
  if (model_client->LoadFromFile(model_path.c_str())) {
    VLOG(3) << "[HUAWEI_ASCEND_NPU] Loading model file success:" << model_path;
    return model_client;
  }
  return nullptr;
}

std::mutex Device::device_mutex_;

bool Device::Build(std::vector<char>* model_buffer, const std::string model_cache_dir) {
  std::lock_guard<std::mutex> lock(device_mutex_);

  ge::Graph ir_graph("graph");
  // ir_graph.SetInputs(input_nodes).SetOutputs(output_nodes);
  // if (GenYoloV3Graph(ir_graph)) {
  if (GenConcatGraph(ir_graph)) {
  // if (GenConcatDGraph(ir_graph)) {
    LOG(INFO) << "[HUAWEI_ASCEND_NPU] GenGGenerate YoloV3 IR graph succees";
  } else {
    LOG(ERROR) << "[HUAWEI_ASCEND_NPU] GenGGenerate YoloV3 IR graph failed!";
    return false;
  }

  // Build IR model
  ge::ModelBufferData om_buffer;
  std::map<std::string, std::string> options;
  options.insert(std::make_pair(ge::ir_option::LOG_LEVEL, "error"));

  ATC_CALL(aclgrphBuildModel(ir_graph, options, om_buffer));

  // Copy from om model buffer
  model_buffer->resize(om_buffer.length);
  memcpy(reinterpret_cast<void*>(model_buffer->data()),
         reinterpret_cast<void*>(om_buffer.data.get()),
         om_buffer.length);

  // Cache om buffer to file
  if (!model_cache_dir.empty()) {
    ATC_CALL(ge::aclgrphSaveModel(model_cache_dir, om_buffer));
  }

  return true;
}

void Device::InitOnce() {
  if (runtime_inited_) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] runtime already inited!";
    return;
  }
  // ACL runtime init => can only be called once in one process
  ACL_CALL(aclInit(NULL));

  // ATC builder init => can only be called once in one process
  std::map<std::string, std::string> global_options;
  global_options.insert(
      std::make_pair(ge::ir_option::SOC_VERSION, "Ascend310"));
  ATC_CALL(ge::aclgrphBuildInitialize(global_options));

  runtime_inited_ = true;
}

void Device::DestroyOnce() {
  if (!runtime_inited_) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] no need to destroy runtime!";
    return;
  }
  // ATC builder finalize => can only be called once in one process
  ge::aclgrphBuildFinalize();
  // ACL runtime finalize => can only be called once in one process
  ACL_CALL(aclFinalize());

  runtime_inited_ = false;
}

} // namespace my_lite_demo