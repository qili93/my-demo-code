#include "utility.h"

const std::string& CvtFormat(ge::Format format) {
  static const int MAX_FORMAT_LENGTH = 25;
  static const std::string format2string[] = {"FORMAT_NCHW = 0",
                                              "FORMAT_NHWC = 1",
                                              "FORMAT_ND = 2",
                                              "FORMAT_NC1HWC0 = 3",
                                              "FORMAT_FRACTAL_Z = 4",
                                              "FORMAT_NC1C0HWPAD = 5",
                                              "FORMAT_NHWC1C0 = 6",
                                              "FORMAT_FSR_NCHW = 7",
                                              "FORMAT_FRACTAL_DECONV = 8",
                                              "FORMAT_C1HWNC0 = 9",
                                              "FORMAT_FRACTAL_DECONV_TRANSPOSE = 10",
                                              "FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS = 11",
                                              "FORMAT_NC1HWC0_C04 = 12",
                                              "FORMAT_FRACTAL_Z_C04 = 13",
                                              "FORMAT_CHWN = 14",
                                              "FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = 15",
                                              "FORMAT_HWCN = 16",
                                              "FORMAT_NC1KHKWHWC0 = 17",
                                              "FORMAT_BN_WEIGHT = 18",
                                              "FORMAT_FILTER_HWCK = 19",
                                              "FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20",
                                              "FORMAT_HASHTABLE_LOOKUP_KEYS = 21",
                                              "FORMAT_HASHTABLE_LOOKUP_VALUE = 22",
                                              "FORMAT_HASHTABLE_LOOKUP_OUTPUT = 23",
                                              "FORMAT_HASHTABLE_LOOKUP_HITS = 24"};
  auto x = static_cast<int>(format);
  CHECK_LT(x, MAX_FORMAT_LENGTH);
  return format2string[x];
}

const std::string& CvtDataType(ge::DataType data_type) {
  static const int MAX_DATATYPE_LENGTH =14;
  static const std::string datatype2string[] = {"DT_FLOAT=0",
                                                "DT_FLOAT16=1",
                                                "DT_INT8=2",
                                                "DT_INT32=3",
                                                "DT_UINT8=4",
                                                "Unknown=5",
                                                "DT_INT16=6",
                                                "DT_UINT16=7",
                                                "DT_UINT32=8",
                                                "DT_INT64=9",
                                                "DT_UINT64=10",
                                                "DT_DOUBLE=11",
                                                "DT_BOOL=12",
                                                "DT_STRING=13"};
                                                
  auto x = static_cast<int>(data_type);
  CHECK_LT(x, MAX_DATATYPE_LENGTH);
  return datatype2string[x];
}

void DebugGeTensorDescInfo(const std::string& name, ge::TensorDesc tensor_desc) {
  VLOG(3) << "[HUAWEI_ASCEND] <" << name << "> Tensor Desc Name is " << tensor_desc.GetName();
  VLOG(3) << "[HUAWEI_ASCEND] <" << name << "> Tensor Desc Format is " << CvtFormat(tensor_desc.GetFormat());
  VLOG(3) << "[HUAWEI_ASCEND] <" << name << "> Tensor Desc DataType is " << CvtDataType(tensor_desc.GetDataType());
  VLOG(3) << "[HUAWEI_ASCEND] <" << name << "> Tensor Desc Origin Format is " << CvtFormat(tensor_desc.GetOriginFormat());
  VLOG(3) << "[HUAWEI_ASCEND] <" << name << "> Tensor Desc Shape GetDimNum is " << tensor_desc.GetShape().GetDimNum();
  VLOG(3) << "[HUAWEI_ASCEND] <" << name << "> Tensor Desc Shape GetShapeSize is " << tensor_desc.GetShape().GetShapeSize();
  for (size_t j = 0; j < tensor_desc.GetShape().GetDimNum(); j++) {
    VLOG(3) << "[HUAWEI_ASCEND] <" << name << "> Tensor Desc Shape [" << j <<"] is " << tensor_desc.GetShape().GetDim(j);
  }
  for (size_t j = 0; j < tensor_desc.GetOriginShape().GetDimNum(); j++) {
    VLOG(3) << "[HUAWEI_ASCEND] <" << name << "> Tensor Desc Origin Shape [" << j <<"] is " << tensor_desc.GetOriginShape().GetDim(j);
  }
}

void DebugGeTensorInfo(const std::string & name, ge::Tensor *ge_tensor) {
  DebugGeTensorDescInfo(name, ge_tensor->GetTensorDesc());
  size_t data_size = reinterpret_cast<size_t>(ge_tensor->GetSize() / sizeof(float));
  float* data_ptr = reinterpret_cast<float*>(ge_tensor->GetData());
  for (size_t index = 0; index < data_size; index++) {
    VLOG(3) << "[HUAWEI_ASCEND] <" << name <<"> Tensor Data [" << index << "]=" << data_ptr[index];
  }
}

std::string to_string(int index) {
  const int BUFFER_LENGTH = 15;
  char buffer[BUFFER_LENGTH];
  snprintf(buffer, sizeof(buffer), "%d", index);
  return std::string(buffer);
}

void DebugGeOPInfo(const std::string& op_name, ge::Operator *ge_op) {
  VLOG(3) << "[HUAWEI_ASCEND] ===========================Start Debuging GE Operator Info===========================";
  VLOG(3) << "[HUAWEI_ASCEND] [" << op_name << "] OP Name is " << ge_op->GetName();
  VLOG(3) << "[HUAWEI_ASCEND] [" << op_name << "] OP Type is " << ge_op->GetOpType();

  if (ge_op->GetOpType() == "Const") {
    ge::Tensor const_tensor;
    ge_op->GetAttr("value", const_tensor);
    DebugGeTensorInfo(op_name+" Const", &const_tensor);
  }
  
  VLOG(3) << "[HUAWEI_ASCEND] ------Print all input info ...";
  for (size_t i = 0; i < ge_op->GetInputsSize(); i++)
  {
    DebugGeTensorDescInfo(op_name+" Input ["+to_string(i)+"]", ge_op->GetInputDesc(i));
  }
  VLOG(3) << "[HUAWEI_ASCEND] ------Print all output info ...";
  for (size_t i = 0; i < ge_op->GetOutputsSize(); i++)
  {
    DebugGeTensorDescInfo(op_name+" Output ["+to_string(i)+"]", ge_op->GetOutputDesc(i));
  }
  VLOG(3) << "[HUAWEI_ASCEND] ------Print all attr info ...";
  const std::map<std::string, std::string> attr_list = ge_op->GetAllAttrNamesAndTypes();
  for (auto iter = attr_list.begin(); iter != attr_list.end(); iter++) {
    VLOG(3) << "[HUAWEI_ASCEND] " << op_name << " Attr [" << iter->first <<"] is " << iter->second;
  }
  VLOG(3) << "[HUAWEI_ASCEND] ===========================Finish Debuging GE Operator Info===========================";
}