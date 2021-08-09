#include "acl/acl.h"

void InitDevice(const int device_id) {
  auto status1 = aclInit(nullptr);
  std::cout << "Call aclInit, status = " << status1 << std::endl;
  auto status2 = aclrtSetDevice(device_id);
  std::cout << "Call aclrtSetDevice, status = " << status2 << std::endl;
}

void Finalize(const int device_id) {
  auto status1 = aclrtResetDevice(device_id);
  std::cout << "Call aclrtResetDevice, status = " << status1 << std::endl;
  auto status2 = aclFinalize();
  std::cout << "Call aclFinalize, status = " << status2 << std::endl;
}

void CreateContext(aclrtContext& context) {
  auto status = aclrtGetCurrentContext(&context);
  std::cout << "Call aclrtGetCurrentContext, status = " << status << std::endl;
}

void DestroyContext(aclrtContext& context) {
  auto status = aclrtDestroyContext(context);
  std::cout << "Call aclrtDestroyContext, status = " << status << std::endl;
}

void CreateStream(aclrtStream& stream) {
  auto status = aclrtCreateStream(&stream);
  std::cout << "Call aclrtCreateStream, status = " << status << std::endl;
}

void WaitStream(aclrtStream& stream) {
  auto status = aclrtSynchronizeStream(&stream);
  std::cout << "Call aclrtSynchronizeStream, status = " << status << std::endl;
}

void DestroyStream(aclrtStream& stream) {
  if (stream) {
    WaitStream(stream);
    auto status = aclrtDestroyStream(stream);
    std::cout << "Call aclrtDestroyStream, status = " << status << std::endl;
  }
}


