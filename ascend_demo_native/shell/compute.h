#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "model_client.h"

class DeviceProgram {
 public:
  DeviceProgram() {}
  ~DeviceProgram() {}
  bool LoadFromCacheFile(const std::string& model_cache_dir);
  bool InitDeivceTensors(std::vector<std::shared_ptr<ge::Tensor>>& device_itensors,
                 std::vector<std::shared_ptr<ge::Tensor>>& device_otensors);  
  bool BuildGraphAndCacheToFile(ge::Graph& om_graph, const std::string& model_cache_dir);
  bool ZeroCopyRun(std::vector<std::shared_ptr<ge::Tensor>>* device_itensors,
                   std::vector<std::shared_ptr<ge::Tensor>>* device_otensors);

 public:
  std::string model_name_{""};
  std::shared_ptr<AclModelClient> model_client_{nullptr};
  std::vector<std::vector<int64_t>> origin_odims_;
  std::vector<TensorDesc> device_idims_{};
  std::vector<TensorDesc> device_odims_{};
};