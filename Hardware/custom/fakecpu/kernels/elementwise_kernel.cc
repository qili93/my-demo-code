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
void AddGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   const phi::DenseTensor& dout,
                   int axis,
                   phi::DenseTensor* dx,
                   phi::DenseTensor* dy) {
  // auto x_data = x.data<T>();
  // auto y_data = y.data<T>();
  // auto out_data = dout.data<T>();
  if (dx) {
    auto dx_data = dev_ctx.template Alloc<T>(dx);
    auto x_numel = dx->numel();
    for (auto i = 0; i < x_numel; ++i) {
      dx_data[i] = static_cast<T>(1.0);
    }
  }
  if (dy) {
    auto dy_data = dev_ctx.template Alloc<T>(dy);
    auto y_numel = dy->numel();
    for (auto i = 0; i < y_numel; ++i) {
      dy_data[i] = static_cast<T>(1.0);
    }
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

PD_REGISTER_PLUGIN_KERNEL(add_grad,
                          custom_cpu,
                          ALL_LAYOUT,
                          custom_kernel::AddGradKernel,
                          int32_t,
                          int64_t,
                          float,
                          double) {}
