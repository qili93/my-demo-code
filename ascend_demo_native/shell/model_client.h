#pragma once

#include <vector>
#include <memory>
#include "utils.h"
#include "logging.h"
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
    if (format == ACL_FORMAT_NHWC) {
      dim_order[1] = 3;
      dim_order[2] = 1;
      dim_order[3] = 2;
    }
    // create ge::Tensordesc
    ge_tensor_desc_ = new ge::TensorDesc(GetGeShape(dims), GetGeFormat(format), GetGeDataType(data_type));
    CHECK(ge_tensor_desc_ != nullptr);
  }
  ~TensorDesc() {
    ge_tensor_desc_ = nullptr;
  }
  int64_t GetNumber() const {
    return ge_tensor_desc_->GetShape().GetDim(dim_order[0]);
  }
  int64_t GetChannel() const {
    return ge_tensor_desc_->GetShape().GetDim(dim_order[1]);
  }
  int64_t GetHeight() const {
    return ge_tensor_desc_->GetShape().GetDim(dim_order[2]);
  }
  int64_t GetWidth() const {
    return ge_tensor_desc_->GetShape().GetDim(dim_order[3]);
  }
  const ge::TensorDesc& GetGeTensorDesc() const { return *ge_tensor_desc_; }

 private:
  ge::Shape GetGeShape(aclmdlIODims dims) {
    ge::Shape ge_shape({0, 0, 0, 0});
    for (size_t i = 0; i < dims.dimCount; i++) {
      if (ge_shape.SetDim(i, dims.dims[i]) != ge::GRAPH_SUCCESS) {
        LOG(WARNING) << "[ASCEND] ge::Shape SetDim failed!";
      }
      else {
        VLOG(3) << "[ASCEND] Setting Ge Shape["<< i <<"] = <" << dims.dims[i] <<">";
      }
    }
    return ge_shape;
  }
  ge::Format GetGeFormat(aclFormat format) {
    ge::Format ge_format = ge::FORMAT_NCHW;
    switch (format) {
      case ACL_FORMAT_NCHW:
        ge_format = ge::FORMAT_NCHW;
        break;
      case ACL_FORMAT_NHWC:
        ge_format = ge::FORMAT_NHWC;
        break;
      case ACL_FORMAT_ND:
        ge_format = ge::FORMAT_ND;
        break;
      default:
        LOG(FATAL) << "[ASCEND] format not supported:" << format;
        break;
    }
    return ge_format;
  }
  ge::DataType GetGeDataType(aclDataType data_type) {
    ge::DataType ge_datatype = ge::DT_FLOAT;
    switch (data_type) {
      case ACL_FLOAT:
        ge_datatype = ge::DT_FLOAT;
        break;
      case ACL_FLOAT16:
        ge_datatype = ge::DT_FLOAT16;
        break;
      case ACL_INT8:
        ge_datatype = ge::DT_INT8;
        break;
      case ACL_INT16:
        ge_datatype = ge::DT_INT16;
        break;
      case ACL_INT32:
        ge_datatype = ge::DT_INT32;
        break;
      case ACL_INT64:
        ge_datatype = ge::DT_INT64;
        break;
      case ACL_BOOL:
        ge_datatype = ge::DT_BOOL;
        break;
      default:
        LOG(FATAL) << "[ASCEND] data type not supported!";
        break;
    }
    return ge_datatype;
  }
  
 private:
  ge::TensorDesc* ge_tensor_desc_{nullptr};
  // n c h w order, default to ACL_FORMAT_NCHW
  std::vector<size_t> dim_order{0, 1, 2, 3};
};

class AclModelClient {
 public:
  AclModelClient() {}
  virtual ~AclModelClient() {}

  bool LoadFromMem(const void* data, uint32_t size);
  bool LoadFromFile(const char* model_path);
  bool GetModelIOTensorDim(std::vector<TensorDesc> *input_tensor, std::vector<TensorDesc> *output_tensor);
  bool ModelExecute(std::vector<std::shared_ptr<ge::Tensor>> *input_tensor, std::vector<std::shared_ptr<ge::Tensor>> *output_tensor);
 private:
  aclmdlDataset* CreateInputDataset(std::vector<std::shared_ptr<ge::Tensor>>* input_tensor);
  aclmdlDataset* CreateOutputDataset(std::vector<std::shared_ptr<ge::Tensor>>* output_tensor);
  bool GetTensorFromDataset(aclmdlDataset * output_dataset, std::vector<std::shared_ptr<ge::Tensor>> *output_tensor);
 private:
  uint32_t model_id_{0};
  aclmdlDesc* model_desc_{nullptr};
};