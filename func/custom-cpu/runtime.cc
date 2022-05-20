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

#include <malloc.h>

#include "paddle/phi/backends/device_ext.h" // 自定义Runtime依赖头文件

static size_t global_total_mem_size = 1 * 1024 * 1024 * 1024UL;
static size_t global_free_mem_size = global_total_mem_size;

C_Status set_device(const C_Device device) {
  return C_SUCCESS;
}

C_Status get_device(const C_Device device) {
  device->id = 0;
  return C_SUCCESS;
}

C_Status get_device_count(size_t *count) {
  *count = 1;
  return C_SUCCESS;
}

C_Status get_device_list(size_t *device) {
  *device = 0;
  return C_SUCCESS;
}

C_Status memory_copy(const C_Device device, void *dst, const void *src, size_t size) {
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status allocate(const C_Device device, void **ptr, size_t size) {
  if (size > global_free_mem_size) {
    return C_FAILED;
  }
  global_free_mem_size -= size;
  *ptr = malloc(size);
  return C_SUCCESS;
}

C_Status deallocate(const C_Device device, void *ptr, size_t size) {
  if (!ptr) {
    return C_FAILED;
  }
  global_free_mem_size += size;
  free(ptr);
  return C_SUCCESS;
}

C_Status create_stream(const C_Device device, C_Stream *stream) {
  stream = nullptr;
  return C_SUCCESS;
}

C_Status destroy_stream(const C_Device device, C_Stream stream) {
  return C_SUCCESS;
}

C_Status create_event(const C_Device device, C_Event *event) {
  return C_SUCCESS;
}

C_Status record_event(const C_Device device, C_Stream stream, C_Event event) {
  return C_SUCCESS;
}

C_Status destroy_event(const C_Device device, C_Event event) {
  return C_SUCCESS;
}

C_Status sync_device(const C_Device device) {
  return C_SUCCESS;
}

C_Status sync_stream(const C_Device device, C_Stream stream) {
  return C_SUCCESS;
}

C_Status sync_event(const C_Device device, C_Event event) {
  return C_SUCCESS;
}

C_Status stream_wait_event(const C_Device device, C_Stream stream, C_Event event) {
  return C_SUCCESS;
}

C_Status memstats(const C_Device device, size_t *total_memory, size_t *free_memory) {
  *total_memory = global_total_mem_size;
  *free_memory = global_free_mem_size;
  return C_SUCCESS;
}

C_Status get_min_chunk_size(const C_Device device, size_t *size) {
  *size = 1;
  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  // 将检查版本兼容性并填充插件使用的自定义 Runtime 版本信息
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);

  // 填充 Runtime 基本信息
  params->device_type = "CustomCPU";
  params->sub_device_type = "V1";

  // 注册 Runtime API
  params->interface->set_device = set_device;
  params->interface->get_device = get_device;
  params->interface->create_stream = create_stream;
  params->interface->destroy_stream = destroy_stream;
  params->interface->create_event = create_event;
  params->interface->destroy_event = destroy_event;
  params->interface->record_event = record_event;
  params->interface->synchronize_device = sync_device;
  params->interface->synchronize_stream = sync_stream;
  params->interface->synchronize_event = sync_event;
  params->interface->stream_wait_event = stream_wait_event;
  params->interface->memory_copy_h2d = memory_copy;
  params->interface->memory_copy_d2d = memory_copy;
  params->interface->memory_copy_d2h = memory_copy;
  params->interface->device_memory_allocate = allocate;
  params->interface->device_memory_deallocate = deallocate;
  params->interface->get_device_count = get_device_count;
  params->interface->get_device_list = get_device_list;
  params->interface->device_memory_stats = memstats;
  params->interface->device_min_chunk_size = get_min_chunk_size;
}
