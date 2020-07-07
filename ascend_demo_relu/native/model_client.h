#pragma once

#include <vector>
#include <memory>
#include "utils.h"
#include "acl/acl.h"
#include "graph/graph.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "graph/ge_error_codes.h"
#include "ge/ge_api_types.h"
#include "ge/ge_ir_build.h"

class TensorDesc {
 public:
  TensorDesc(aclDataType data_type, aclmdlIODims dims, aclFormat format) {
    tensor_desc_ =
        aclCreateTensorDesc(data_type, dims.dimCount, dims.dims, format);
    if(tensor_desc_ == nullptr) {
      ERROR_LOG("aclCreateTensorDesc Failed!");
    }
    aclSetTensorDescName(tensor_desc_, dims.name);
    if (format == ACL_FORMAT_NHWC) {
      dim_order[1] = 3;
      dim_order[2] = 1;
      dim_order[3] = 2;
    }
  }
  ~TensorDesc() {
    if (tensor_desc_ != nullptr) {
      aclDestroyTensorDesc(tensor_desc_);
      tensor_desc_ = nullptr;
    }
  }
  uint32_t GetNumber() const {
    return static_cast<uint32_t>(
        aclGetTensorDescDim(tensor_desc_, dim_order[0]));
  }
  uint32_t GetChannel() const {
    return static_cast<uint32_t>(
        aclGetTensorDescDim(tensor_desc_, dim_order[1]));
  }
  uint32_t GetHeight() const {
    return static_cast<uint32_t>(
        aclGetTensorDescDim(tensor_desc_, dim_order[2]));
  }
  uint32_t GetWidth() const {
    return static_cast<uint32_t>(
        aclGetTensorDescDim(tensor_desc_, dim_order[3]));
  }
  const aclTensorDesc& GetTensorDesc() const { return *tensor_desc_; }

 private:
  aclTensorDesc* tensor_desc_{nullptr};
  // n c h w order, default to ACL_FORMAT_NCHW
  std::vector<uint32_t> dim_order{0, 1, 2, 3};
};

class AclModelClient {
 public:
  AclModelClient() {}
  virtual ~AclModelClient() {}

  bool LoadFromMem(const void* data, uint32_t size);
  bool LoadFromFile(const char* model_path);
  //bool SaveModelToFile(const std::string& filename, const std::vector<char>& contents);
  bool GetModelIOTensorDim(std::vector<TensorDesc>& input_tensor, std::vector<TensorDesc>& output_tensor);
  bool ModelExecute(std::vector<std::shared_ptr<ge::Tensor>> &input_tensor, std::vector<std::shared_ptr<ge::Tensor>> &output_tensor);
 private:
  aclmdlDataset * CreateDatasetFromTensor(std::vector<std::shared_ptr<ge::Tensor>> &input_tensor, bool is_input);
  bool GetTensorFromDataset(aclmdlDataset * output_dataset, std::vector<std::shared_ptr<ge::Tensor>> &output_tensor);
  //bool CreateOutputDataset(std::vector<std::shared_ptr<ge::Tensor>> &output_tensor);
 private:
  uint32_t model_id_{0};
  aclmdlDesc* model_desc_{nullptr};
};