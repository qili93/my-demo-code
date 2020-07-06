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
  if (model_buffer.size() == 0) {
    return nullptr;
  }
  // Init resources before run model
  if (!InitDevice()) {
    ERROR_LOG("[ASCEND] InitDevice failed!.");
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
  std::ifstream fs(model_path);
  if (!fs.is_open()) {
    ERROR_LOG("[device](LoadFromFile) model file <%s> not exists!", model_path.c_str());
    return nullptr;
  }
  // Init resources before run model
  if (!InitDevice()) {
    ERROR_LOG("[ASCEND] InitDevice failed!.");
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
//   ge::Graph ir_graph("graph");
//   ir_graph.SetInputs(input_nodes).SetOutputs(output_nodes);

//   // System init
//   std::map<std::string, std::string> global_options = {
//       {ge::ir_option::SOC_VERSION, "Ascend310"},
//   };
//   auto status = ge::aclgrphBuildInitialize(global_options);
//   if (status != ge::GRAPH_SUCCESS) {
//     ERROR_LOG("[ASCEND] InitDevice failed!.");
//     return false;
//   }
//   // Build IR model
//   ge::ModelBufferData om_buffer;
//   std::map<std::string, std::string> options;
//   // Do nothing?
//   status = ge::aclgrphBuildModel(ir_graph, options, om_buffer);
//   if (status != ge::GRAPH_SUCCESS) {
//     ERROR_LOG("[ASCEND] aclgrphBuildModel failed!");
//   }
//   INFO_LOG("[ASCEND] aclgrphBuildModel success.");

//   // Copy from om model buffer
//   model_buffer->resize(om_buffer.length);
//   memcpy(reinterpret_cast<uint8_t*>(model_buffer->data()),
//          reinterpret_cast<uint8_t*>(om_buffer.data.get()),
//          om_buffer.length);

//   // release resource
//   ge::aclgrphBuildFinalize();
//   INFO_LOG("[ASCEND] Build model done.");
//   return true;
// }

bool Device::InitDevice() {
  // skip if device already inited
  if (device_inited_) return true;

  // ACL init
  aclError ret = aclInit(NULL);
  if (ret != ACL_ERROR_NONE) {
    ERROR_LOG("[device](InitDevice) aclInit failed!");
    return false;
  }
  INFO_LOG("[device](InitDevice) aclInit succeed!");
  // Open Device
  ret = aclrtSetDevice(device_id_);
  if (ret != ACL_ERROR_NONE) {
    ERROR_LOG("[device](InitDevice) acl open device %d failed", device_id_);
    return false;
  }
  INFO_LOG("[device](InitDevice) open device %d success", device_id_);
  // create context (set current)
  ret = aclrtCreateContext(&context_, device_id_);
  if (ret != ACL_ERROR_NONE) {
    ERROR_LOG("[device](InitDevice) acl create context failed");
    return false;
  }
  INFO_LOG("[device](InitDevice) create context success");
  // create stream
  ret = aclrtCreateStream(&stream_);
  if (ret != ACL_ERROR_NONE) {
    ERROR_LOG("[device](InitDevice) acl create stream failed");
    return false;
  }
  INFO_LOG("[device](InitDevice) create stream success");
  // get run mode
  aclrtRunMode runMode;
  ret = aclrtGetRunMode(&runMode);
  if (ret != ACL_ERROR_NONE) {
    ERROR_LOG("[device](InitDevice) acl get run mode failed");
    return false;
  }
  runmode_is_device_ = (runMode == ACL_DEVICE);
  INFO_LOG("[device](InitDevice) get run mode success");

  device_inited_ = true;
  return true;
}

void Device::ReleaseDevice() {
  // skip if device not inited
  if (!device_inited_) return;

  aclError ret;
  if (stream_ != nullptr) {
    ret = aclrtDestroyStream(stream_);
    if (ret != ACL_ERROR_NONE) {
      ERROR_LOG("[device](ReleaseDevice) destroy stream failed");
    }
    stream_ = nullptr;
  }
  INFO_LOG("[device](ReleaseDevice) end to destroy stream");

  if (context_ != nullptr) {
    ret = aclrtDestroyContext(context_);
    if (ret != ACL_ERROR_NONE) {
      ERROR_LOG("[device](ReleaseDevice) destroy context failed");
    }
    context_ = nullptr;
  }
  INFO_LOG("[device](ReleaseDevice) end to destroy context");

  ret = aclrtResetDevice(device_id_);
  if (ret != ACL_ERROR_NONE) {
    ERROR_LOG("[device](ReleaseDevice) reset device failed");
  }
  INFO_LOG("[device](ReleaseDevice) end to reset device is %d", device_id_);

  ret = aclFinalize();
  if (ret != ACL_ERROR_NONE) {
    ERROR_LOG("[device](ReleaseDevice) finalize acl failed");
  }
  INFO_LOG("[device](ReleaseDevice) end to finalize acl");

  device_inited_ = false;
}

} // namespace my_lite_demo