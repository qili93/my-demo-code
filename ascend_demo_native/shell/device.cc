#include <string.h>
#include <iostream>
#include <fstream>
#include "ge/ge_ir_build.h"
#include "ge/ge_api_types.h"
#include "graph/graph.h"
#include "ge/ge_ir_build.h"

#include "device.h"

namespace my_lite_demo {

std::shared_ptr<AclModelClient> Device::LoadFromMem(const std::vector<char>& model_buffer) {
  LOG(INFO) << "[ASCEND] Staring LoadFromMem ...";
  if (model_buffer.size() == 0) {
    return nullptr;
  }
  // Init resources before run model
  if (!InitDevice()) {
    LOG(ERROR) << "[ASCEND] InitDevice failed!.";
    return nullptr;
  }
  // Create a ACL model  client to load the om model
  std::shared_ptr<AclModelClient> model_client(new AclModelClient());
  // Load model from memory
  if (model_client->LoadFromMem(reinterpret_cast<const void*>(model_buffer.data()), model_buffer.size())){
    return model_client;
  }
  return nullptr;
}

std::shared_ptr<AclModelClient> Device::LoadFromFile(const std::string& model_path) {
  LOG(INFO) << "[ASCEND] Staring LoadFromFile ...";
  std::ifstream fs(model_path);
  if (!fs.is_open()) {
    LOG(ERROR) << "[ASCEND] model file " << model_path.c_str() << " not exists!";
    return nullptr;
  }
  // Init resources before run model
  if (!InitDevice()) {
    LOG(ERROR) << "[ASCEND] InitDevice failed!.";
    return nullptr;
  }
  // Create a ACL model  client to load the om model
  std::shared_ptr<AclModelClient> model_client(new AclModelClient());
  // Load model from memory
  if (model_client->LoadFromFile(model_path.c_str())) {
    return model_client;
  }
  return nullptr;
}

// bool Device::Build(std::vector<ge::Operator>& input_nodes,   // NOLINT
//                    std::vector<ge::Operator>& output_nodes,  // NOLINT
//                    std::vector<char>* model_buffer) {
//   // Convert the HiAI IR graph to the HiAI om model
//   ge::Graph ir_graph("graph";
//   ir_graph.SetInputs(input_nodes).SetOutputs(output_nodes);

//   // System init
//   std::map<std::string, std::string> global_options = {
//       {ge::ir_option::SOC_VERSION, "Ascend310"},
//   };
//   auto status = ge::aclgrphBuildInitialize(global_options);
//   if (status != ge::GRAPH_SUCCESS) {
//     LOG(ERROR) << "[ASCEND] InitDevice failed!.";
//     return false;
//   }
//   // Build IR model
//   ge::ModelBufferData om_buffer;
//   std::map<std::string, std::string> options;
//   // Do nothing?
//   status = ge::aclgrphBuildModel(ir_graph, options, om_buffer);
//   if (status != ge::GRAPH_SUCCESS) {
//     LOG(ERROR) << "[ASCEND] aclgrphBuildModel failed!";
//   }
//   LOG(INFO) << "[ASCEND] aclgrphBuildModel success.";

//   // Copy from om model buffer
//   model_buffer->resize(om_buffer.length);
//   memcpy(reinterpret_cast<uint8_t*>(model_buffer->data()),
//          reinterpret_cast<uint8_t*>(om_buffer.data.get()),
//          om_buffer.length);

//   // release resource
//   ge::aclgrphBuildFinalize();
//   LOG(INFO) << "[ASCEND] Build model done.";
//   return true;
// }

bool Device::InitDevice() {
  LOG(INFO) << "[ASCEND] Starting InitDevice ...";
  // skip if device already inited
  if (device_inited_) return true;

  // ACL init
  aclError ret = aclInit(NULL);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[ASCEND] aclInit failed!";
    return false;
  }
  LOG(INFO) << "[ASCEND] aclInit succeed!";
  // Open Device
  ret = aclrtSetDevice(device_id_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[ASCEND] acl open device " << device_id_ << " failed";
    return false;
  }
  LOG(INFO) << "[ASCEND] open device " << device_id_ << " success";
  // create context (set current)
  ret = aclrtCreateContext(&context_, device_id_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[ASCEND] acl create context failed";
    return false;
  }
  LOG(INFO) << "[ASCEND] create context success";
  // create stream
  ret = aclrtCreateStream(&stream_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[ASCEND] acl create stream failed";
    return false;
  }
  LOG(INFO) << "[ASCEND] create stream success";
  // get run mode
  aclrtRunMode runMode;
  ret = aclrtGetRunMode(&runMode);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[ASCEND] acl get run mode failed";
    return false;
  }
  runmode_is_device_ = (runMode == ACL_DEVICE);
  LOG(INFO) << "[ASCEND] get run mode success";

  device_inited_ = true;
  LOG(INFO) << "[ASCEND] Finishing InitDevice ...";
  return true;
}

void Device::ReleaseDevice() {
  // skip if device not inited
  if (!device_inited_) return;

  aclError ret;
  if (stream_ != nullptr) {
    ret = aclrtDestroyStream(stream_);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "[ASCEND] destroy stream failed";
    }
    stream_ = nullptr;
  }
  LOG(INFO) << "[ASCEND] end to destroy stream";

  if (context_ != nullptr) {
    ret = aclrtDestroyContext(context_);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "[ASCEND] destroy context failed";
    }
    context_ = nullptr;
  }
  LOG(INFO) << "[ASCEND] end to destroy context";

  ret = aclrtResetDevice(device_id_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[ASCEND] reset device failed";
  }
  LOG(INFO) << "[ASCEND] end to reset device is " << device_id_;

  ret = aclFinalize();
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[ASCEND] finalize acl failed";
  }
  LOG(INFO) << "[ASCEND] end to finalize acl";

  device_inited_ = false;
}

} // namespace my_lite_demo