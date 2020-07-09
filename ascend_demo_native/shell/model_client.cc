
#include "ge/ge_ir_build.h"
#include "ge/ge_api_types.h"
#include "graph/graph.h"
#include "ge/ge_ir_build.h"
#include "model_client.h"

bool AclModelClient::LoadFromMem(const void* data, uint32_t size) {
  auto ret = aclmdlLoadFromMem(data, size, &model_id_);
  if (ret != ACL_ERROR_NONE) {
    LOG(WARNING) << "[ASCEND] Load model from memory failed!";
    return false;
  }
  model_desc_ = aclmdlCreateDesc();
  if (model_desc_ == nullptr) {
    LOG(WARNING) << "[ASCEND] create model description failed!";
    return false;
  }
  ret = aclmdlGetDesc(model_desc_, model_id_);
  if (ret != ACL_ERROR_NONE) {
    LOG(WARNING) << "[ASCEND] get model description failed!";
    return false;
  }
  return true;
}

bool AclModelClient::LoadFromFile(const char* model_path) {
  LOG(INFO) << "[ASCEND] Starting LoadFromFile ...";
  LOG(INFO) << "[ASCEND] model_path is " << model_path;
  auto ret = aclmdlLoadFromFile(model_path, &model_id_);
  if (ret != ACL_ERROR_NONE) {
    LOG(WARNING) << "[ASCEND] Load model from file failed!";
    return false;
  }
  model_desc_ = aclmdlCreateDesc();
  if (model_desc_ == nullptr) {
    LOG(WARNING) << "[ASCEND] create model description failed!";
    return false;
  }
  ret = aclmdlGetDesc(model_desc_, model_id_);
  if (ret != ACL_ERROR_NONE) {
    LOG(WARNING) << "[ASCEND] get model description failed!";
    return false;
  }
  return true;
}

// bool AclModelClient::SaveModelToFile(const std::string& filename,
//                       const std::vector<char>& contents) {
//   ge::ModelBufferData om_buffer;
//   const char* ptr = reinterpret_cast<const char*>(&(contents.at(0)));
//   om_buffer.data= std::make_shared<uint8_t>(*ptr);
//   om_buffer.length = contents.size();
 
//   auto status = ge::aclgrphSaveModel(filename, om_buffer);
//   if (status != ge::GRAPH_SUCCESS) {
//     LOG(WARNING) << "[ASCEND] aclgrphSaveModel failed!";
//     return false;
//   }
//   LOG(INFO) << "[ASCEND] aclgrphSaveModel succeed!";
//   return true;
// }

bool AclModelClient::GetModelIOTensorDim(std::vector<TensorDesc> *input_tensor, std::vector<TensorDesc> *output_tensor) {
  if (!model_desc_) {
    LOG(WARNING) << "[ASCEND] GetModelIOTensorDim failed!";
    return false;
  }
  size_t input_num = aclmdlGetNumInputs(model_desc_);
  for (size_t i = 0; i < input_num; i++) {
    aclmdlIODims input_dim;
    aclmdlGetInputDims(model_desc_, i, &input_dim);
    aclDataType data_type = aclmdlGetInputDataType(model_desc_, i);
    LOG(INFO) << "[ASCEND] data_type of inputs[" << i <<"] is " << data_type;
    aclFormat data_format = aclmdlGetInputFormat(model_desc_, i);
    LOG(INFO) << "[ASCEND] data_format of inputs[" << i <<"] is " << data_format;
    TensorDesc tensor_desc = TensorDesc(data_type, input_dim, data_format);
    input_tensor->push_back(tensor_desc);
  }

  size_t output_num = aclmdlGetNumOutputs(model_desc_);
  for (size_t i = 0; i < output_num; i++) {
    aclmdlIODims output_dim;
    aclmdlGetOutputDims(model_desc_, i, &output_dim);
    aclDataType data_type = aclmdlGetOutputDataType(model_desc_, i);
    LOG(INFO) << "[ASCEND] data_type of outputs[" << i <<"] is " << data_type;
    aclFormat data_format = aclmdlGetOutputFormat(model_desc_, i);
    LOG(INFO) << "[ASCEND] data_format of outputs[" << i <<"] is " << data_format;
    TensorDesc tensor_desc = TensorDesc(data_type, output_dim, data_format);
    output_tensor->push_back(tensor_desc);
  }
  return true;
}

// void* AclModelClient::GetDeviceBufferOfTensor(std::shared_ptr<ge::Tensor> &tensor) {
//   void * device_buffer = nullptr;
//   aclError ret = aclrtMalloc(&device_buffer, tensor->GetSize(), ACL_MEM_MALLOC_NORMAL_ONLY);
//   if (ret != ACL_ERROR_NONE) {
//     LOG(WARNING) << "[ASCEND] aclrtMalloc failed!";
//     return nullptr;
//   }
//   ret = aclrtMemcpy(device_buffer, tensor->GetSize(), item->GetData(), tensor->GetSize(), ACL_MEMCPY_HOST_TO_DEVICE);
//   if (ret != ACL_ERROR_NONE) {
//     LOG(WARNING) << "[ASCEND] aclrtMemcpy failed!";
//     aclrtFree(device_buffer);
//     return nullptr;
//   }
//   return device_buffer;
// }

bool AclModelClient::GetTensorFromDataset(aclmdlDataset * output_dataset, std::vector<std::shared_ptr<ge::Tensor>> *output_tensor) {
  size_t device_output_num = aclmdlGetDatasetNumBuffers(output_dataset);
  size_t tensor_output_num = reinterpret_cast<size_t>(output_tensor->size());
  if (device_output_num != tensor_output_num) {
    LOG(ERROR) << "output number not equal, device number is " << device_output_num << ", tensor number is " << tensor_output_num;
    return false;
  }
  for (size_t i = 0; i < device_output_num; i++) {
    aclDataBuffer* buffer_device = aclmdlGetDatasetBuffer(output_dataset, i);
    void * device_data = aclGetDataBufferAddr(buffer_device);
    uint32_t device_size = aclGetDataBufferSize(buffer_device);
    LOG(INFO) << "[ASCEND] buffer size of output dataset is " << device_size;

    void* tensor_data = nullptr;
    aclError ret = aclrtMallocHost(&tensor_data, device_size);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "[ASCEND] aclrtMallocHost failed, ret " << ret;
      return false;
    }
    LOG(INFO) << "[ASCEND] succeed to malloc host memory for output.";
    ret = aclrtMemcpy(tensor_data, device_size, device_data, device_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "[ASCEND] aclrtMemcpy failed, ret " << ret;
      return false;
    }
    LOG(INFO) << "[ASCEND] succeed to memory copy from device to host.";
    if (output_tensor->at(i)->SetData(reinterpret_cast<uint8_t*>(tensor_data), device_size) != ge::GRAPH_SUCCESS) {
      LOG(ERROR) << "[ASCEND] SetData to output tensor failed";
      return false;
    }
    LOG(INFO) << "[ASCEND] succeed set host memory data to output tensor.";
  }
  LOG(INFO) << "[ASCEND] Get output tensor from output dataset succeed.";
  return true;
}

aclmdlDataset* AclModelClient::CreateInputDataset(std::vector<std::shared_ptr<ge::Tensor>>* input_tensor) {
  aclmdlDataset* dataset = aclmdlCreateDataset();
  if (dataset == nullptr) {
    LOG(ERROR) << "[ASCEND] can't create dataset, create output failed!";
    return nullptr;
  }
  for (size_t i = 0; i < input_tensor->size(); i++) {
    auto item = input_tensor->at(i);
    size_t buffer_size = item->GetSize();
    void * buffer_device = nullptr;
    aclError ret = aclrtMalloc(&buffer_device, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
      //LOG(ERROR) << "malloc device buffer failed. size is %lu", buffer_size);
      LOG(ERROR) << "[ASCEND] input malloc device buffer failed. size is " << buffer_size;
      return nullptr;
    }
    void * buffer_data = reinterpret_cast<void*>(item->GetData());
    ret = aclrtMemcpy(buffer_device, buffer_size, buffer_data, buffer_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "[ASCEND] memcpy failed. device buffer size is " << buffer_size << " input host buffer size is " << buffer_size;
      aclrtFree(buffer_device);
      return nullptr;
    }
    aclDataBuffer* data_buffer = aclCreateDataBuffer(buffer_device, buffer_size);
    if (aclmdlAddDatasetBuffer(dataset, data_buffer) != ACL_ERROR_NONE) {
      LOG(WARNING) << "[ASCEND] aclmdlAddDatasetBuffer failed!";
      aclDestroyDataBuffer(data_buffer);
      return nullptr;
    }
  }
  LOG(INFO) << "[ASCEND] CreateInputDataset succeed.";
  return dataset;
}

aclmdlDataset* AclModelClient::CreateOutputDataset(std::vector<std::shared_ptr<ge::Tensor>>* output_tensor) {
  if (model_desc_ == nullptr) {
    LOG(ERROR) << "[ASCEND] no model description, create ouput failed!";
    return nullptr;
  }
  aclmdlDataset* dataset = aclmdlCreateDataset();
  if (dataset == nullptr) {
    LOG(ERROR) << "[ASCEND] can't create dataset, create output failed!";
    return nullptr;
  }
  size_t output_size = aclmdlGetNumOutputs(model_desc_);
  LOG(INFO) << "[ASCEND] output_size is " << output_size;
  CHECK_EQ(output_size, output_tensor->size());
  for (size_t i = 0; i < output_size; i ++) {
    size_t buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);
    LOG(INFO) << "[ASCEND] output_size["<< i <<"] is " << buffer_size;
    LOG(INFO) << "[ASCEND] output_size["<< i <<"] of float is " << (buffer_size / sizeof(float));
    void* buffer_device = nullptr;
    aclError ret = aclrtMalloc(&buffer_device, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "[ASCEND] output malloc device buffer failed. size is "
                 << buffer_size;
      return nullptr;
    }
    aclDataBuffer* data_buffer = aclCreateDataBuffer(buffer_device, buffer_size);
    if (aclmdlAddDatasetBuffer(dataset, data_buffer) != ACL_ERROR_NONE) {
      LOG(ERROR) << "[ASCEND] output aclmdlAddDatasetBuffer failed!";
      aclrtFree(buffer_device);
      aclDestroyDataBuffer(data_buffer);
      return nullptr;
    }
  }
  LOG(INFO) << "[ASCEND] CreateOutputDataset succeed.";
  return dataset;
}

bool AclModelClient::ModelExecute(std::vector<std::shared_ptr<ge::Tensor>> *input_tensor, 
                                  std::vector<std::shared_ptr<ge::Tensor>> *output_tensor) {

  // print input_tensor
  for (size_t i = 0; i < input_tensor->size(); i++) {
    auto item = input_tensor->at(i);
    size_t input_size = reinterpret_cast<size_t>(item->GetSize() / sizeof(float));
    float* input_data = reinterpret_cast<float*>(item->GetData());
    for (size_t index = 0; index < input_size; index++) {
      LOG(INFO) << "[ASCEND] input_tensor[" << index << "]=" << input_data[index];
    }
  }

  // create input/output dataset
  aclmdlDataset *input_dataset = CreateInputDataset(input_tensor);
  aclmdlDataset *output_dataset = CreateOutputDataset(output_tensor);

  // model execution
  aclError ret = aclmdlExecute(model_id_, input_dataset, output_dataset);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[ASCEND] aclmdlExecute failed, modelId: " << model_id_;
    return false;
  }
  LOG(INFO) << "[ASCEND] aclmdlExecute succeed, modelId: " << model_id_;

  // get output
  if (!GetTensorFromDataset(output_dataset, output_tensor)) {
    LOG(ERROR) << "[ASCEND] GetTensorFromDataset failed, modelId: " << model_id_;
  }
  LOG(INFO) << "[ASCEND] GetTensorFromDataset succeed, modelId: " << model_id_;

  // print output tensor
  for (size_t i = 0; i < output_tensor->size(); i++) {
    auto item = output_tensor->at(i);
    size_t output_size = reinterpret_cast<size_t>(item->GetSize() / sizeof(float));
    float* output_data = reinterpret_cast<float*>(item->GetData());
    for (size_t index = 0; index < output_size; index++) {
      LOG(INFO) << "[ASCEND] output_tensor[" << index << "]=" << output_data[index];
    }
  }
  return true;
}