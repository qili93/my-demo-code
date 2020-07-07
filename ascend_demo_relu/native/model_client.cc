
#include "ge/ge_ir_build.h"
#include "ge/ge_api_types.h"
#include "graph/graph.h"
#include "ge/ge_ir_build.h"
#include "model_client.h"

bool AclModelClient::LoadFromMem(const void* data, uint32_t size) {
  auto ret = aclmdlLoadFromMem(data, size, &model_id_);
  if (ret != ACL_ERROR_NONE) {
    WARN_LOG("[model_client] Load model from memory failed!");
    return false;
  }
  model_desc_ = aclmdlCreateDesc();
  if (model_desc_ == nullptr) {
    WARN_LOG("[model_client] create model description failed!");
    return false;
  }
  ret = aclmdlGetDesc(model_desc_, model_id_);
  if (ret != ACL_ERROR_NONE) {
    WARN_LOG("[model_client] get model description failed!");
    return false;
  }
  return true;
}

bool AclModelClient::LoadFromFile(const char* model_path) {
  INFO_LOG("-------Enter AclModelClient::LoadFromFile -------");
  INFO_LOG("model_path=<%s>", model_path);
  auto ret = aclmdlLoadFromFile(model_path, &model_id_);
  if (ret != ACL_ERROR_NONE) {
    WARN_LOG("[model_client] Load model from file failed!");
    return false;
  }
  model_desc_ = aclmdlCreateDesc();
  if (model_desc_ == nullptr) {
    WARN_LOG("[model_client] create model description failed!");
    return false;
  }
  ret = aclmdlGetDesc(model_desc_, model_id_);
  if (ret != ACL_ERROR_NONE) {
    WARN_LOG("[model_client] get model description failed!");
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
//     WARN_LOG("[model_client] aclgrphSaveModel failed!");
//     return false;
//   }
//   INFO_LOG("[model_client] aclgrphSaveModel succeed!");
//   return true;
// }

bool AclModelClient::GetModelIOTensorDim(std::vector<TensorDesc>& input_tensor, std::vector<TensorDesc>& output_tensor) {
  if (!model_desc_) {
    WARN_LOG("[model_client] GetModelIOTensorDim failed!");
    return false;
  }
  size_t input_num = aclmdlGetNumInputs(model_desc_);
  for (size_t i = 0; i < input_num; i++)
  {
    //size_t input_size = aclmdlGetInputSizeByIndex(model_desc_, i);
    aclmdlIODims input_dim;
    aclmdlGetInputDims(model_desc_, i, &input_dim);
    aclDataType data_type = aclmdlGetInputDataType(model_desc_, i);
    aclFormat data_format = aclmdlGetInputFormat(model_desc_, i);
    TensorDesc tensor_desc = TensorDesc(data_type, input_dim, data_format);
    input_tensor.push_back(tensor_desc);
  }

  size_t output_num = aclmdlGetNumOutputs(model_desc_);
  for (size_t i = 0; i < output_num; i++)
  {
    //size_t input_size = aclmdlGetOutputSizeByIndex(model_desc_, i);
    aclmdlIODims output_dim;
    aclmdlGetOutputDims(model_desc_, i, &output_dim);
    aclDataType data_type = aclmdlGetOutputDataType(model_desc_, i);
    aclFormat data_format = aclmdlGetOutputFormat(model_desc_, i);
    TensorDesc tensor_desc = TensorDesc(data_type, output_dim, data_format);
    output_tensor.push_back(tensor_desc);
  }
  return true;
}

// void* AclModelClient::GetDeviceBufferOfTensor(std::shared_ptr<ge::Tensor> &tensor) {
//   void * device_buffer = nullptr;
//   aclError ret = aclrtMalloc(&device_buffer, tensor->GetSize(), ACL_MEM_MALLOC_NORMAL_ONLY);
//   if (ret != ACL_ERROR_NONE) {
//     WARN_LOG("[model_client] aclrtMalloc failed!");
//     return nullptr;
//   }
//   ret = aclrtMemcpy(device_buffer, tensor->GetSize(), item->GetData(), tensor->GetSize(), ACL_MEMCPY_HOST_TO_DEVICE);
//   if (ret != ACL_ERROR_NONE) {
//     WARN_LOG("[model_client] aclrtMemcpy failed!");
//     aclrtFree(device_buffer);
//     return nullptr;
//   }
//   return device_buffer;
// }

bool AclModelClient::GetTensorFromDataset(aclmdlDataset * output_dataset, std::vector<std::shared_ptr<ge::Tensor>> &output_tensor) {
  size_t device_output_num = aclmdlGetDatasetNumBuffers(output_dataset);
  size_t tensor_output_num = reinterpret_cast<size_t>(output_tensor.size());
  if (device_output_num != tensor_output_num) {
    ERROR_LOG("output number not equal, device number is %lu, tensor number is %lu", device_output_num, tensor_output_num);
    return false;
  }
  for (size_t i = 0; i < device_output_num; i++) {
    aclDataBuffer* buffer_device = aclmdlGetDatasetBuffer(output_dataset, i);
    void * device_data = aclGetDataBufferAddr(buffer_device);
    uint32_t device_size = aclGetDataBufferSize(buffer_device);

    void * tensor_data = reinterpret_cast<void*>(output_tensor[i]->GetData());
    size_t tensor_size = output_tensor[i]->GetSize();
    if((static_cast<size_t>(device_size)) != tensor_size) {
      ERROR_LOG("index <%lu> size not equal, device szie is %u, tensor size is %lu", i, device_size, tensor_size);
      return false;
    }
    aclError ret = aclrtMemcpy(tensor_data, tensor_size, device_data, device_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
      ERROR_LOG("aclrtMemcpy failed, ret[%d]", ret);
      return false;
    }
    //tensor_data = reinterpret_cast<float*>(tensor_data);
  }
  return true;
}

aclmdlDataset *AclModelClient::CreateDatasetFromTensor(std::vector<std::shared_ptr<ge::Tensor>> &tensor, bool is_input) {
  aclmdlDataset *dataset = aclmdlCreateDataset();
  for (auto item : tensor) {
    size_t buffer_size = item->GetSize();
    void * buffer_device = nullptr;
    aclError ret = aclrtMalloc(&buffer_device, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
      ERROR_LOG("malloc device buffer failed. size is %lu", buffer_size);
      return nullptr;
    }
    if (is_input) {
      void * buffer_data = reinterpret_cast<void*>(item->GetData());
      ret = aclrtMemcpy(buffer_device, buffer_size, buffer_data, buffer_size, ACL_MEMCPY_HOST_TO_DEVICE);
      if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("memcpy failed. device buffer size is %lu, input host buffer size is %lu", buffer_size, buffer_size);
        aclrtFree(buffer_device);
        return nullptr;
      }
    }
    aclDataBuffer* data_buffer = aclCreateDataBuffer(buffer_device, buffer_size);
    if (aclmdlAddDatasetBuffer(dataset, data_buffer) != ACL_ERROR_NONE) {
      WARN_LOG("[model_client] aclmdlAddDatasetBuffer failed!");
      aclDestroyDataBuffer(data_buffer);
      return nullptr;
    }
  }
  return dataset;
}

// bool AclModelClient::CreateOutputDataset(std::vector<std::shared_ptr<ge::Tensor>> &output_tensor) {
//   return true;
// }

bool AclModelClient::ModelExecute(std::vector<std::shared_ptr<ge::Tensor>> &input_tensor, std::vector<std::shared_ptr<ge::Tensor>> &output_tensor) {
  // print input_tensor
  for (auto item : input_tensor) {
    size_t input_size = reinterpret_cast<size_t>(item->GetSize() / sizeof(float));
    float *input_data = reinterpret_cast<float*>(item->GetData());
    for (size_t index = 0; index < input_size; index++) {
      INFO_LOG("[model_client](ModelExecute) input_tensor[%lu]=%f", index, input_data[index]);
    }
  }
  // // print output tensor
  // for (auto item : output_tensor) {
  //   size_t input_size = item->GetSize();
  //   uint8_t *input_data = item->GetData();
  //   for (size_t index = 0; index < input_size; index++) {
  //     INFO_LOG("[model_client](ModelExecute) output_tensor[%d]=<5d>", index, input_data[index]);
  //   }
  // }

  // create input/output dataset
  aclmdlDataset *input_dataset = CreateDatasetFromTensor(input_tensor, true);
  aclmdlDataset *output_dataset = CreateDatasetFromTensor(output_tensor, false);

  // model execution
  aclError ret = aclmdlExecute(model_id_, input_dataset, output_dataset);
  if (ret != ACL_ERROR_NONE) {
    ERROR_LOG("[model_client](ModelExecute) aclmdlExecute failed, modelId: %d", model_id_);
    return false;
  }
  INFO_LOG("[model_client](ModelExecute) aclmdlExecute succeed, modelId: %d", model_id_);

  // get output
  if (!GetTensorFromDataset(output_dataset, output_tensor)) {
    ERROR_LOG("[model_client](ModelExecute) GetTensorFromDataset failed, modelId %d", model_id_);
  }
  INFO_LOG("[model_client](ModelExecute) GetTensorFromDataset succeed, modelId %d", model_id_);

  // print output tensor
  for (auto item : output_tensor) {
    size_t output_size = reinterpret_cast<size_t>(item->GetSize() / sizeof(float));
    float *output_data = reinterpret_cast<float*>(item->GetData());
    for (size_t index = 0; index < output_size; index++) {
      INFO_LOG("[model_client](ModelExecute) output_tensor[%lu]=%f", index, output_data[index]);
    }
  }
  return true;
}