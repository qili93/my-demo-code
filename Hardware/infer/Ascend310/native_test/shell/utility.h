#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "logging.h"
#include "acl/acl.h"
#include "ge/ge_api_types.h"
#include "ge/ge_ir_build.h"
#include "graph/ge_error_codes.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "all_ops.h" // opp/op_proto/built-in/inc

/*
 * This file contains some Huawei Ascend NPU specific uitls.
 */


#define TENSOR_UPDATE_INPUT(op, attr, format, dtype)                    \
  ge::TensorDesc _##op##_input_desc_##attr(ge::Shape(), format, dtype); \
  op.update_input_desc_##attr(_##op##_input_desc_##attr);
#define TENSOR_UPDATE_OUTPUT(op, attr, format, dtype)                    \
  ge::TensorDesc _##op##_output_desc_##attr(ge::Shape(), format, dtype); \
  op.update_output_desc_##attr(_##op##_output_desc_##attr);
#define TENSOR_UPDATE_DYNAMIC_INPUT(op, attr, idx, format, dtype)               \
  ge::TensorDesc _##op##_input_desc_##attr##_##idx(ge::Shape(), format, dtype); \
  op.update_dynamic_input_desc_##attr(idx, _##op##_input_desc_##attr##_##idx);
#define TENSOR_UPDATE_DYNAMIC_OUTPUT(op, attr, idx, format, dtype)               \
  ge::TensorDesc _##op##_output_desc_##attr##_##idx(ge::Shape(), format, dtype); \
  op.update_dynamic_output_desc_##attr(idx, _##op##_output_desc_##attr##_##idx);

std::string to_string(int index);

const std::string& CvtFormat(ge::Format format);
const std::string& CvtDataType(ge::DataType data_type);

void DebugGeTensorDescInfo(const std::string& name, ge::TensorDesc tensor_desc);
void DebugGeTensorInfo(const std::string & name, ge::Tensor *ge_tensor);
void DebugGeOPInfo(const std::string& op_name, ge::Operator *ge_op);

#define ACL_CALL(msg)                                       \
  CHECK_EQ(reinterpret_cast<aclError>(msg), ACL_ERROR_NONE) \
      << (msg) << " Huawei Ascend NPU ACL Error: "          \
      << ::paddle::lite::huawei_ascend_npu::AclErrorInfo(   \
             reinterpret_cast<int>(msg))

#define ATC_CALL(msg)                                                 \
  CHECK_EQ(reinterpret_cast<ge::graphStatus>(msg), ge::GRAPH_SUCCESS) \
      << (msg) << " Huawei Ascend NPU ATC Error: "                    \
      << ::paddle::lite::huawei_ascend_npu::AtcErrorInfo(             \
             reinterpret_cast<uint32_t>(msg))

namespace paddle {
namespace lite {
namespace huawei_ascend_npu {

static const char* AtcErrorInfo(uint32_t error) {
  switch (error) {
#define LITE_ATC_ERROR_INFO(xx) \
  case xx:                      \
    return #xx;                 \
    break;
    LITE_ATC_ERROR_INFO(ge::GRAPH_FAILED);         // 0xFFFFFFFF
    LITE_ATC_ERROR_INFO(ge::GRAPH_PARAM_INVALID);  // 50331649
#undef LITE_ATC_ERROR_INFO
    default:
      return "unknown error";
      break;
  }
}

static const char* AclErrorInfo(int error) {
  switch (error) {
#define LITE_ACL_ERROR_INFO(xx) \
  case xx:                      \
    return #xx;                 \
    break;
    LITE_ACL_ERROR_INFO(ACL_ERROR_INVALID_PARAM);                    // 100000
    LITE_ACL_ERROR_INFO(ACL_ERROR_UNINITIALIZE);                     // 100001
    LITE_ACL_ERROR_INFO(ACL_ERROR_REPEAT_INITIALIZE);                // 100002
    LITE_ACL_ERROR_INFO(ACL_ERROR_INVALID_FILE);                     // 100003
    LITE_ACL_ERROR_INFO(ACL_ERROR_WRITE_FILE);                       // 100004
    LITE_ACL_ERROR_INFO(ACL_ERROR_INVALID_FILE_SIZE);                // 100005
    LITE_ACL_ERROR_INFO(ACL_ERROR_PARSE_FILE);                       // 100006
    LITE_ACL_ERROR_INFO(ACL_ERROR_FILE_MISSING_ATTR);                // 100007
    LITE_ACL_ERROR_INFO(ACL_ERROR_FILE_ATTR_INVALID);                // 100008
    LITE_ACL_ERROR_INFO(ACL_ERROR_INVALID_DUMP_CONFIG);              // 100009
    LITE_ACL_ERROR_INFO(ACL_ERROR_INVALID_PROFILING_CONFIG);         // 100010
    LITE_ACL_ERROR_INFO(ACL_ERROR_INVALID_MODEL_ID);                 // 100011
    LITE_ACL_ERROR_INFO(ACL_ERROR_DESERIALIZE_MODEL);                // 100012
    LITE_ACL_ERROR_INFO(ACL_ERROR_PARSE_MODEL);                      // 100013
    LITE_ACL_ERROR_INFO(ACL_ERROR_READ_MODEL_FAILURE);               // 100014
    LITE_ACL_ERROR_INFO(ACL_ERROR_MODEL_SIZE_INVALID);               // 100015
    LITE_ACL_ERROR_INFO(ACL_ERROR_MODEL_MISSING_ATTR);               // 100016
    LITE_ACL_ERROR_INFO(ACL_ERROR_MODEL_INPUT_NOT_MATCH);            // 100017
    LITE_ACL_ERROR_INFO(ACL_ERROR_MODEL_OUTPUT_NOT_MATCH);           // 100018
    LITE_ACL_ERROR_INFO(ACL_ERROR_MODEL_NOT_DYNAMIC);                // 100019
    LITE_ACL_ERROR_INFO(ACL_ERROR_OP_TYPE_NOT_MATCH);                // 100020
    LITE_ACL_ERROR_INFO(ACL_ERROR_OP_INPUT_NOT_MATCH);               // 100021
    LITE_ACL_ERROR_INFO(ACL_ERROR_OP_OUTPUT_NOT_MATCH);              // 100022
    LITE_ACL_ERROR_INFO(ACL_ERROR_OP_ATTR_NOT_MATCH);                // 100023
    LITE_ACL_ERROR_INFO(ACL_ERROR_OP_NOT_FOUND);                     // 100024
    LITE_ACL_ERROR_INFO(ACL_ERROR_OP_LOAD_FAILED);                   // 100025
    LITE_ACL_ERROR_INFO(ACL_ERROR_UNSUPPORTED_DATA_TYPE);            // 100026
    LITE_ACL_ERROR_INFO(ACL_ERROR_FORMAT_NOT_MATCH);                 // 100027
    LITE_ACL_ERROR_INFO(ACL_ERROR_BIN_SELECTOR_NOT_REGISTERED);      // 100028
    LITE_ACL_ERROR_INFO(ACL_ERROR_KERNEL_NOT_FOUND);                 // 100029
    LITE_ACL_ERROR_INFO(ACL_ERROR_BIN_SELECTOR_ALREADY_REGISTERED);  // 100030
    LITE_ACL_ERROR_INFO(ACL_ERROR_KERNEL_ALREADY_REGISTERED);        // 100031
    LITE_ACL_ERROR_INFO(ACL_ERROR_INVALID_QUEUE_ID);                 // 100032
    LITE_ACL_ERROR_INFO(ACL_ERROR_REPEAT_SUBSCRIBE);                 // 100033
    LITE_ACL_ERROR_INFO(ACL_ERROR_STREAM_NOT_SUBSCRIBE);             // 100034
    LITE_ACL_ERROR_INFO(ACL_ERROR_THREAD_NOT_SUBSCRIBE);             // 100035
    LITE_ACL_ERROR_INFO(ACL_ERROR_WAIT_CALLBACK_TIMEOUT);            // 100036
    LITE_ACL_ERROR_INFO(ACL_ERROR_REPEAT_FINALIZE);                  // 100037
    LITE_ACL_ERROR_INFO(ACL_ERROR_NOT_STATIC_AIPP);                  // 100038
    LITE_ACL_ERROR_INFO(ACL_ERROR_BAD_ALLOC);                        // 200000
    LITE_ACL_ERROR_INFO(ACL_ERROR_API_NOT_SUPPORT);                  // 200001
    LITE_ACL_ERROR_INFO(ACL_ERROR_INVALID_DEVICE);                   // 200002
    LITE_ACL_ERROR_INFO(ACL_ERROR_MEMORY_ADDRESS_UNALIGNED);         // 200003
    LITE_ACL_ERROR_INFO(ACL_ERROR_RESOURCE_NOT_MATCH);               // 200004
    LITE_ACL_ERROR_INFO(ACL_ERROR_INVALID_RESOURCE_HANDLE);          // 200005
    LITE_ACL_ERROR_INFO(ACL_ERROR_FEATURE_UNSUPPORTED);              // 200006
    LITE_ACL_ERROR_INFO(ACL_ERROR_STORAGE_OVER_LIMIT);               // 300000
    LITE_ACL_ERROR_INFO(ACL_ERROR_INTERNAL_ERROR);                   // 500000
    LITE_ACL_ERROR_INFO(ACL_ERROR_FAILURE);                          // 500001
    LITE_ACL_ERROR_INFO(ACL_ERROR_GE_FAILURE);                       // 500002
    LITE_ACL_ERROR_INFO(ACL_ERROR_RT_FAILURE);                       // 500003
    LITE_ACL_ERROR_INFO(ACL_ERROR_DRV_FAILURE);                      // 500004
    LITE_ACL_ERROR_INFO(ACL_ERROR_PROFILING_FAILURE);                // 500005
#undef LITE_ACL_ERROR_INFO
    default:
      return "unknown error";
      break;
  }
}

}  // namespace huawei_ascend_npu
}  // namespace lite
}  // namespace paddle