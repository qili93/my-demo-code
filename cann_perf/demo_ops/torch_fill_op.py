# Load in relevant libraries, and alias where appropriate
import time
import argparse
import datetime
import torch
import torch.npu

out = torch.full(size=(4096, 1000), fill_value=1, dtype=torch.int64, device="npu:0")
print(out)


# aclopCompileAndExecute: <Fills>, Inputs:{[ACL_INT64,(4096,1000),ACL_FORMAT_ND],}, Outputs:{[ACL_INT64,(4096,1000),ACL_FORMAT_ND],}

# tensor([[1, 1, 1,  ..., 1, 1, 1],
#         [1, 1, 1,  ..., 1, 1, 1],
#         [1, 1, 1,  ..., 1, 1, 1],
#         ...,
#         [1, 1, 1,  ..., 1, 1, 1],
#         [1, 1, 1,  ..., 1, 1, 1],
#         [1, 1, 1,  ..., 1, 1, 1]], device='npu:0')
# THPModule_npu_shutdown success.
