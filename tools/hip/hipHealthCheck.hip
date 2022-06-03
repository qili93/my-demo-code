#include <hip/hip_runtime.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <cstdlib>

#define HIP_SUCCESS(err) if (err != hipSuccess) std::cout \
      << "HIP func:" << __FUNCTION__ << "failed at line: " << __LINE__  \
      <<  ", got an error: " << hipGetErrorString(err) << std::endl // NOLINT(*)

int main(int argc, char* argv[])
{
  // Get Device Count
  int count;
  HIP_SUCCESS(hipGetDeviceCount(&count));
  std::cout << "Visiable GPU count = " << count << std::endl;

  // set device id based on input
  int device_id = 0;
  if (argc > 1) {
    device_id = atoi(argv[1]);
  }
    if (device_id < 0 || device_id > count - 1) {
    std::cout << "Invalid device_id: " << device_id << ", force device_id to <0>" << std::endl;
  }
  // Set Device to device_id
  HIP_SUCCESS(hipSetDevice(device_id));
  std::cout << "Set device id to: " << device_id << std::endl;

  // Get GPU Memory of device_id
  size_t actual_avail = 0;
  size_t actual_total = 0;
  HIP_SUCCESS(hipMemGetInfo(&actual_avail, &actual_total));
  std::cout << "actual_avail = " << actual_avail << ", actual_total = " << actual_total << std::endl;

  // Get Device Properties of device_id
  hipDeviceProp_t devProp;
  HIP_SUCCESS(hipGetDeviceProperties(&devProp, device_id));
  std::cout << " devProp.name = " << devProp.name << std::endl;
  std::cout << " devProp.gcnArch = " << devProp.gcnArch << std::endl;
  std::cout << " devProp.gcnArchName = " << devProp.gcnArchName << std::endl;
  std::cout << " devProp.major = " << devProp.major << std::endl;
  std::cout << " devProp.minor = " << devProp.minor << std::endl;
  std::cout << " devProp.multiProcessorCount = " << devProp.multiProcessorCount << std::endl;
  std::cout << " devProp.clockRate = " << devProp.clockRate << std::endl;

  // Get TFLOPS
  int sm_per_multiproc = 64 * 2;
  unsigned long long compute_perf  = (unsigned long long) devProp.multiProcessorCount * sm_per_multiproc * devProp.clockRate;
  std::cout << " compute_perf = " << compute_perf << std::endl;
  float tflops = (float)compute_perf / 100000000;
  std::cout <<setiosflags(std::ios::fixed)<<std::setprecision(2);
  std::cout << " tflops = " << tflops << " TFLOPS" << std::endl;

  return 0;
}

