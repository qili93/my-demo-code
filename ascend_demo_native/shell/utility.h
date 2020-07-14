#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "logging.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "graph/operator.h"

std::string to_string(int index);

const std::string& CvtFormat(ge::Format format);
const std::string& CvtDataType(ge::DataType data_type);

void DebugGeTensorDescInfo(const std::string& name, ge::TensorDesc tensor_desc);
void DebugGeTensorInfo(const std::string & name, ge::Tensor *ge_tensor);
void DebugGeOPInfo(const std::string& op_name, ge::Operator *ge_op);