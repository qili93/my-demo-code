#include <unistd.h>
#include "model_build.h"
#include "model_client.h"
#include "device.h"
#include "subgraph_compute.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    LOG(ERROR) <<  "[ERROR] usage: ./" << argv[0] << " model_dir\n";
    exit(1);
  }
  std::string model_dir = argv[1];
  VLOG(3) << "[main] Getting model dir to " << model_dir;

  auto device_program = std::make_shared<DeviceProgram>(1);
  if (device_program->LoadFromCacheFile(model_dir+".om")) {
    LOG(INFO) << "[main] LoadFromCacheFile succees";
  } else {
    device_program->BuildGraphAndCacheToFile(model_dir);
  }

  std::vector<std::shared_ptr<ge::Tensor>> device_itensors_{};
  std::vector<std::shared_ptr<ge::Tensor>> device_otensors_{};
  if (device_program->InitDeivceTensors(device_itensors_, device_otensors_)) {
    LOG(INFO) << "[main] InitDeivceTensors succees";
  } else {
    LOG(ERROR) << "[main] InitDeivceTensors failed!";
  }

  if (device_program->ZeroCopyRun(&device_itensors_, &device_otensors_)) {
    LOG(INFO) << "[main] ZeroCopyRun succees";
  } else {
    LOG(ERROR) << "[main] ZeroCopyRun failed!";
  }
  return 0;
}
