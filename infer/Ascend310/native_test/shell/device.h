#pragma once

#include <string>
#include <memory>
#include <mutex>
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
  Device() { InitOnce(); }

  ~Device() { DestroyOnce(); }

  // Load the ir om model from buffer, and create a ACL model client to run
  // inference
  std::shared_ptr<AclModelClient> LoadFromMem(
      const std::vector<char>& model_buffer, const int device_id);
  std::shared_ptr<AclModelClient> LoadFromFile(const std::string& model_path,
                                               const int device_id);
  // Build the ACL IR graph to the ACL om model
  bool Build(std::vector<char>* model_buffer, const std::string model_cache_dir);

  bool InitDevice();
  bool ReleaseDevice();

 private:
  void InitOnce();
  void DestroyOnce();
  bool runtime_inited_{false};
  static std::mutex device_mutex_;
};

} // namespace my_lite_demo