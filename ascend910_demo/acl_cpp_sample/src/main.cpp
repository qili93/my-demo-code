/**
* @file main.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "acl/acl.h"
#include "op_runner.h"

#include "common.h"

bool g_isDevice = false;

OperatorDesc CreateOpDesc()
{
    // define operator
    std::vector<int64_t> shape{2, 3};
    std::string opType = "Add";
    aclDataType dataType = ACL_FLOAT;
    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc(opType);
    opDesc.AddInputTensorDesc(dataType, shape.size(), shape.data(), format);
    opDesc.AddInputTensorDesc(dataType, shape.size(), shape.data(), format);
    opDesc.AddOutputTensorDesc(dataType, shape.size(), shape.data(), format);
    return opDesc;
}

template <typename T>
bool SetInputData(OpRunner &runner)
{
    for (size_t i = 0; i < runner.NumInputs(); ++i) {
        auto input_size = runner.GetInputSize(i) / sizeof(T);
        auto input_data = runner.GetInputBuffer<T>(i);
        for (size_t i = 0; i < input_size; ++i) {
            input_data[i] = static_cast<T>(i);
        }
        INFO_LOG("Input[%zu]:", i);
        runner.PrintInput(i);
        printf("\n");
    }

    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    for (size_t i = 0; i < runner.NumOutputs(); ++i) {
        INFO_LOG("Output[%zu]:", i);
        runner.PrintOutput(i);
        printf("\n");
    }
    return true;
}

template <typename T>
bool RunAddOp()
{
    // [TODO] create op desc
    OperatorDesc opDesc = CreateOpDesc();

    // [TODO] create Runner
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    // [TODO] load inputs
    if (!SetInputData<T>(opRunner)) {
        return false;
    }

    // [TODO] run op
    if (!opRunner.RunOp()) {
        return false;
    }

    // [TODO] process output data
    if (!ProcessOutputData(opRunner)) {
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS) {
        ERROR_LOG("Init acl failed");
        return FAILED;
    }
    int deviceId = 0;
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        std::cerr << "Open device failed. device id = " << deviceId << std::endl;
        return FAILED;
    }
    INFO_LOG("Open device[%d] success", deviceId);

    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("g_isDevice = %d", g_isDevice);

    if (!RunAddOp<float>()) {
        (void) aclrtResetDevice(deviceId);
        return FAILED;
    }

    (void) aclrtResetDevice(deviceId);
    return SUCCESS;
}
