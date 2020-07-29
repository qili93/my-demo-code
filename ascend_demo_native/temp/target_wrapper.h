// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <mutex>  //NOLINT
#include <utility>
#include <vector>
#include "utils.h"

class TargetWrapperHuaweiAscendNPU {
 public:
  using context_t = aclrtContext;
  using stream_t = aclrtStream;
  using event_t = aclrtEvent;

  static size_t num_devices();
  static size_t maximum_stream() { return 1024; }

  static void CreateDevice(int device_id);
  static void DestroyDevice(int device_id);
  static int GetCurDevice() { return device_id_; }

 public:
  static void InitOnce();
  static void DestroyOnce();

 private:
  static bool runtime_inited_;
  static std::vector<int> device_list_;
  static std::mutex device_mutex_;

  static thread_local int device_id_;
};
