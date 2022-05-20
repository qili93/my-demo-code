#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "device.h"

class DeviceProgram {
 public:
  DeviceProgram(int device_id) { device_id_ = device_id; }
  ~DeviceProgram() {}
  bool LoadFromCacheFile(const std::string& model_cache_dir);
  bool InitDeivceTensors(std::vector<std::shared_ptr<ge::Tensor>>& device_itensors,
                 std::vector<std::shared_ptr<ge::Tensor>>& device_otensors);  
  bool BuildGraphAndCacheToFile(const std::string& model_cache_dir);
  bool ZeroCopyRun(std::vector<std::shared_ptr<ge::Tensor>>* device_itensors,
                   std::vector<std::shared_ptr<ge::Tensor>>* device_otensors);

 public:
  int device_id_{0};
  std::string model_name_{""};
  std::shared_ptr<AclModelClient> model_client_{nullptr};
  std::vector<std::vector<int64_t>> origin_odims_;
  std::vector<TensorDesc> device_idims_{};
  std::vector<TensorDesc> device_odims_{};
};