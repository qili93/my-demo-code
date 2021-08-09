#include "acl/acl.h"
#include "acl/acl_op_compiler.h"

aclTensorDesc* CreateTensorDesc(const aclDataType dtype, const aclFormat format, const std::vector<int64_t> dims) {
  int size = dims.size();
  auto *desc = aclCreateTensorDesc(dtype, size, dims.data(), format);
  aclSetTensorStorageFormat(desc, format);
  aclSetTensorStorageShape(desc, size, dims.data());
  return desc;
}
