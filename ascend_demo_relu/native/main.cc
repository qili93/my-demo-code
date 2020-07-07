#include <unistd.h>
#include "model_build.h"
#include "model_client.h"
#include "device.h"
#include "compute.h"

int main(int argc, char* argv[])
{
  std::string model_cache_dir = "../assets/models/native_model";

  ge::Graph graph1("IrGraph1");
  OMModelBuild * om_build = new OMModelBuild();
  if (!om_build->GenGraph(graph1)) {
    ERROR_LOG("[model_build] GenGraph failed!");
  } else {
    INFO_LOG("[model_build] GenGraph succees");
  }

  auto device_program = std::make_shared<DeviceProgram>();
  if (device_program->LoadFromCacheFile(model_cache_dir+".om")) {
    INFO_LOG("[main] LoadFromCacheFile succees");
  } else {
    device_program->BuildGraphAndCacheToFile(graph1, model_cache_dir);
  }
  if (device_program->model_client_ == nullptr) {
    ERROR_LOG("[main] BuildGraphAndCacheToFile failed!");
    return -1;
  }

  std::vector<std::shared_ptr<ge::Tensor>> device_itensors_{};
  std::vector<std::shared_ptr<ge::Tensor>> device_otensors_{};
  if (device_program->InitDeivceTensors(device_itensors_, device_otensors_)) {
    INFO_LOG("[main] InitDeivceTensors succees");
  } else {
    ERROR_LOG("[main] InitDeivceTensors failed!");
  }

  if (device_program->ZeroCopyRun(&device_itensors_, &device_otensors_)) {
    INFO_LOG("[main] ZeroCopyRun succees");
  } else {
    ERROR_LOG("[main] ZeroCopyRun failed!");
  }
  return 0;
}
