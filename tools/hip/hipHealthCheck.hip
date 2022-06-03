#include <hip/hip_runtime.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <cstdlib>

#define HIP_SUCCESS(err) if (err != hipSuccess) std::cout \
    << "ERROR: HIP failed at line: " << __LINE__  <<  " with hip error " \
    << hipGetErrorString(err) << std::endl // NOLINT(*)

int main(int argc, char* argv[])
{
  // Get Device Count
  int count;
  HIP_SUCCESS(hipGetDeviceCount(&count));
  std::cout << "hipGetDeviceCount=<" << count << ">"<< std::endl;

  // set device id based on input
  int device_id = 0;
  if (argc > 1) {
    device_id = atoi(argv[1]);
  }
  if (device_id < 0 || device_id > count - 1) {
    device_id = 0;
    std::cout << "Invalid device_id: " << device_id << ", force device_id to <0>" << std::endl;
  }
  // Set Device to device_id
  HIP_SUCCESS(hipSetDevice(device_id));
  std::cout << "hipSetDevice=<" << device_id << ">" << std::endl;

  // Get GPU Memory of device_id
  size_t avail = 0;
  size_t total = 0;
  HIP_SUCCESS(hipMemGetInfo(&avail, &total));
  std::cout << "hipMemGetInfo: free=<" << avail << ">, total=<" << total << ">" << std::endl;

  // Get Device Properties of device_id
  hipDeviceProp_t devProp;
  HIP_SUCCESS(hipGetDeviceProperties(&devProp, device_id));
  std::cout << "DeviceProp.name=<" << devProp.name << ">" << std::endl;
  std::cout << "DeviceProp.gcnArch=<" << devProp.gcnArch << ">" << std::endl;
  std::cout << "DeviceProp.gcnArchName=<" << devProp.gcnArchName << ">" << std::endl;
  // std::cout << " DeviceProp.major=<" << devProp.major << ">" << std::endl;
  // std::cout << " DeviceProp.minor=<" << devProp.minor << ">" << std::endl;
  // std::cout << " DeviceProp.multiProcessorCount=<" << devProp.multiProcessorCount << ">" << std::endl;
  // std::cout << " DeviceProp.clockRate <= " << devProp.clockRate << ">" << std::endl;

  // Get TFLOPS
  int sm_per_multiproc = 64 * 2;
  unsigned long long compute_perf  = (unsigned long long) devProp.multiProcessorCount * sm_per_multiproc * devProp.clockRate;
  // std::cout << " DeviceProp.compute_perf=<" << compute_perf << ">" << std::endl;
  float tflops = (float)compute_perf / 100000000;
  std::cout <<setiosflags(std::ios::fixed)<<std::setprecision(2);
  std::cout << "DeviceProp.tflops=<" << tflops << "> TFLOPS" << std::endl;

  return 0;
}

