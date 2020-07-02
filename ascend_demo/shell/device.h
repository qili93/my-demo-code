#pragma once

#include <string>
#include <vector>
#include "graph/graph.h"
#include "model_client.h"

namespace my_lite_demo {

class Device {
 public:
  static Device& Global() {
    static Device x;
    return x;
  }
  Device() : device_inited_(false) {}
  ~Device() { ReleaseDevice(); }

  // Load the ir om model from buffer, and create a ACL model client to run
  // inference
  std::shared_ptr<AclModelClient> LoadFromMem(const std::vector<char>& model_buffer);
  std::shared_ptr<AclModelClient> LoadFromFile(const std::string& model_path);
  // Build the ACL IR graph to the ACL om model
  bool Build(std::vector<ge::Operator>& input_nodes,   // NOLINT
             std::vector<ge::Operator>& output_nodes,  // NOLINT
             std::vector<char>* model_buffer);         // NOLINT

  bool test();
  bool runModeIsDevice() const { return runmode_is_device_; }

  bool InitDevice();
  void ReleaseDevice();

 private:
  bool device_inited_{false};
  bool runmode_is_device_{false};
  int32_t device_id_{1};
  aclrtContext context_{nullptr};
  aclrtStream stream_{nullptr};
};

} // namespace my_lite_demo