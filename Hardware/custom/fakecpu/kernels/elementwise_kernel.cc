// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  auto x_data = x.data<T>();
  auto y_data = y.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto numel = out->numel();
  for (auto i = 0; i < numel; ++i) {
    out_data[i] = x_data[i] * y_data[i];
  }
}

template <typename T, typename Context>
void GradAddKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   phi::DenseTensor* out) {
  auto x_data = x.data<T>();
  auto y_data = y.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto numel = out->numel();
  for (auto i = 0; i < numel; ++i) {
    out_data[i] = x_data[i] * y_data[i];
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(add,
                          custom_cpu,
                          ALL_LAYOUT,
                          custom_kernel::AddKernel,
                          int32_t,
                          int64_t,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(grad_add,
                          custom_cpu,
                          ALL_LAYOUT,
                          custom_kernel::GradAddKernel,
                          int32_t,
                          int64_t,
                          float,
                          double) {}
