#include <iostream>
#include <vector>
#include "acl_resource.h"
#include "acl_operator.h"

int main() {
  const std::vector<int64_t> dims{3, 2};
  const std::vector<float> x1{1, 2, 3, 4, 5, 6};
  const std::vector<float> x2{1, 2, 3, 4, 5, 6};
  std::vector<float> y{0, 0, 0, 0, 0, 0};

  InitDevice(0);
  aclrtContext context;
  CreateContext(context);
  aclrtStream stream;
  CreateStream(stream);

  auto x1_desc = CreateTensorDesc(ACL_FLOAT, ACL_FORMAT_ND, dims);
  auto x1_size = aclGetTensorDescSize(x1_desc);
  void* x1_device_ptr;
  aclrtMalloc(&x1_device_ptr,x1_size,ACL_MEM_MALLOC_NORMAL_ONLY);
  aclrtMemcpy(x1_device_ptr, x1_size, x1.data(), x1_size, ACL_MEMCPY_HOST_TO_DEVICE);
  auto x1_buffer = CreateDataBuffer(x1_device_ptr, x1_size);

  auto x2_desc = CreateTensorDesc(ACL_FLOAT, ACL_FORMAT_ND, dims);
  auto x2_size = aclGetTensorDescSize(x1_desc);
  void* x2_device_ptr;
  aclrtMalloc(&x2_device_ptr,x2_size,ACL_MEM_MALLOC_NORMAL_ONLY);
  aclrtMemcpy(x2_device_ptr, x2_size, x2.data(), x2_size, ACL_MEMCPY_HOST_TO_DEVICE);
  auto x2_buffer = CreateDataBuffer(x2_device_ptr, x2_size);

  auto y_desc = CreateTensorDesc(ACL_FLOAT, ACL_FORMAT_ND, dims);
  auto y_size = aclGetTensorDescSize(y_desc);
  void* y_device_ptr;
  aclrtMalloc(&y_device_ptr,y_size,ACL_MEM_MALLOC_NORMAL_ONLY);
  auto y_buffer = CreateDataBuffer(y_device_ptr, y_size);

  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(x1_desc);
  input_descs.emplace_back(x2_desc);
  input_buffers.emplace_back(x1_buffer);
  input_buffers.emplace_back(x2_buffer);

  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(y_desc);
  output_buffers.emplace_back(y_buffer);

  auto attr = CreateAttr();

  aclError ret = aclopCompileAndExecute("Add", input_descs.size(), input_descs.data(),
      input_buffers.data(), output_descs.size(), output_descs.data(),
      output_buffers.data(), attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL,
      stream);

  WaitStream(stream);

  aclrtMemcpy(y.data(), y_size, y_device_ptr, y_size, ACL_MEMCPY_DEVICE_TO_HOST);

  for (int i = 0; i < y.size(); ++i) {
    std::cout << "y[" << i << "] = " << y[i] << std::endl;
  }

  aclDestroyDataBuffer(x1_buffer);
  aclDestroyDataBuffer(x2_buffer);
  aclDestroyDataBuffer(y_buffer);
  aclDestroyTensorDesc(x1_desc);
  aclDestroyTensorDesc(x2_desc);
  aclDestroyTensorDesc(y_desc);
  aclrtFree(x1_device_ptr);
  aclrtFree(x2_device_ptr);
  aclrtFree(y_device_ptr);

  aclopDestroyAttr(attr);

  // release
  DestroyStream(stream);
  DestroyContext(context);
  Finalize(0);

  return 0;
}