/**
* @file sample_process.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "sample_process.h"
#include "utils.h"
using namespace std;

static bool runFlag = true;

SampleProcess::SampleProcess():deviceId_(0), context_(nullptr),
                               outFolder_((char*)"outdir/"), picDesc_({}),
                               format_(2), enType_(0)
{
}

SampleProcess::~SampleProcess()
{
    DestroyResource();
}
void *ThreadFunc(void *arg)
{
    // Notice: create context for this thread
    int deviceId = 0;
    aclrtContext context = nullptr;
    aclError ret = aclrtCreateContext(&context, deviceId);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclrtCreateContext failed, ret=%d.", ret);
        return ((void*)(-1));
    }

    INFO_LOG("thread start ");
    while (runFlag) {
        // Notice: timeout 1000ms
        aclError aclRet = aclrtProcessReport(1000);
    }

    ret = aclrtDestroyContext(context);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclrtDestroyContext failed, ret=%d.", ret);
    }

    return (void*)0;
}

Result SampleProcess::InitResource()
{
    // ACL init
    const char *aclConfigPath = "../src/acl.json";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl init failed");
        return FAILED;
    }
    INFO_LOG("acl init success");

    // open device
    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl open device %d failed", deviceId_);
        return FAILED;
    }
    INFO_LOG("open device %d success", deviceId_);

    // create context (set current)
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create context failed");
        return FAILED;
    }
    INFO_LOG("create context success");

    // create stream
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create stream failed");
        return FAILED;
    }
    INFO_LOG("create stream success");

    // get run mode
    aclrtRunMode runMode;
    ret = aclrtGetRunMode(&runMode);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }
    bool isDivece = (runMode == ACL_DEVICE);
    RunStatus::SetDeviceStatus(isDivece);
    INFO_LOG("get run mode success");

    return SUCCESS;
}

Result SampleProcess::DoVdecProcess()
{
    // create threadId
    int createThreadErr = pthread_create(&threadId_, nullptr, ThreadFunc, nullptr);
    if (createThreadErr != 0) {
        ERROR_LOG("create thread failed, err = %d", createThreadErr);
        return FAILED;
    }
    INFO_LOG("create thread successfully, threadId = %lu", threadId_);
    (void)aclrtSubscribeReport(static_cast<uint64_t>(threadId_), stream_);

    Result ret = Utils::CheckFolder(outFolder_);
    if (ret != SUCCESS) {
        ERROR_LOG("mkdir out folder error.");
        (void)aclrtUnSubscribeReport(static_cast<uint64_t>(threadId_), stream_);
        return FAILED;
    }

    // dvpp init
    VdecProcess processVdec;
    ret = processVdec.InitResource(threadId_, enType_, format_);
    if (ret != SUCCESS) {
        ERROR_LOG("init dvpp resource failed");
        (void)aclrtUnSubscribeReport(static_cast<uint64_t>(threadId_), stream_);
        return FAILED;
    }

    std::string filePath= "../vdec_h265_1frame_rabbit_1280x720.h265";
    const int inputWidth = 1280;
    const int inputHeight = 720;
    int rest_len = 10;
    picDesc_.width = inputWidth;
    picDesc_.height = inputHeight;

    int32_t count = 0;
    while (rest_len > 0) {
        void *inBufferDev = nullptr;
        uint32_t inBufferSize = 0;

        // read file to device memory
        if (!Utils::ReadFileToDeviceMem(filePath.c_str(), inBufferDev, inBufferSize)) {
            ERROR_LOG("read file %s to device mem failed.\n", filePath.c_str());
            continue;
        }
        processVdec.SetInput(inBufferDev, inBufferSize, picDesc_.width, picDesc_.height);

        ret = processVdec.Process();
        if (ret != SUCCESS) {
            ERROR_LOG("dvpp ProcessVdec failed");
            continue;
        }
        ++count;
        rest_len = rest_len - 1;
        INFO_LOG("success to aclvdecSendFrame, count=%d", count);
    }
    processVdec.DestroyResource();

    return SUCCESS;
}

Result SampleProcess::DoModelProcess()
{
    // model init
    ModelProcess processModel;
    const char* omModelPath = "../model/resnet50_aipp.om";
    std::string modelOutputBinfileName = "./result/model_output_";
    std::string dvppOutputfileName = "./result/dvpp_output_";
    Result ret = Utils::CheckFolder("result");
    if (ret != SUCCESS) {
        ERROR_LOG("mkdir out folder error.");
        return FAILED;
    }
    ret = processModel.LoadModelFromFileWithMem(omModelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("execute LoadModelFromFileWithMem failed");
        return FAILED;
    }

    ret = processModel.CreateDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateDesc failed");
        return FAILED;
    }

    ret = processModel.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateOutput failed");
        return FAILED;
    }

    // dvpp init
    DvppProcess processDvpp(stream_);
    ret = processDvpp.InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("init dvpp resource failed");
        return FAILED;
    }

    const int modelInputWidth = 224; // cur model shape is 224 * 224
    const int modelInputHeight = 224;
    std::vector<std::string> fileList = Utils::readDir(outFolder_);
    if(fileList.size() > 0) {
        for(size_t frameId = 0; frameId < fileList.size(); frameId++) {
            void *dvppOutputBuffer = nullptr;
            uint32_t dvppOutputSize;
            // read file to device memory
            std::string fileNameSave = outFolder_ + fileList[frameId];
            if (!Utils::ReadFileToDeviceMem(fileNameSave.c_str(), dvppOutputBuffer, dvppOutputSize)) {
                ERROR_LOG("read file %s to device mem failed.\n", fileNameSave.c_str());
                return FAILED;
            }
            processDvpp.SetInput(picDesc_.width, picDesc_.height, format_);
            ret = processDvpp.InitOutputPara(modelInputWidth, modelInputHeight);
            if (ret != SUCCESS) {
                ERROR_LOG("init dvpp output para failed");
                return FAILED;
            }
            processDvpp.Process(dvppOutputBuffer, dvppOutputSize);

            processDvpp.GetOutput(&dvppOutputBuffer, dvppOutputSize);

            // model proces
            ret = processModel.CreateInput(dvppOutputBuffer, dvppOutputSize);
            if (ret != SUCCESS) {
                ERROR_LOG("execute CreateInput failed");
                acldvppFree(dvppOutputBuffer);
                return FAILED;
            }
            ret = processModel.Execute();
            if (ret != SUCCESS) {
                ERROR_LOG("execute inference failed");
                acldvppFree(dvppOutputBuffer);
                return FAILED;
            }
            acldvppFree(dvppOutputBuffer);
            remove(fileNameSave.c_str());

            aclmdlDataset *modelOutput = processModel.GetModelOutputData();
            if (modelOutput == nullptr) {
                ERROR_LOG("get model output data failed");
                return FAILED;
            }
            std::string modelOutputBinfileNameCur = modelOutputBinfileName + std::to_string(frameId);
            ret = Utils::PullModelOutputData(modelOutput, modelOutputBinfileNameCur.c_str());
            if (ret != SUCCESS) {
                ERROR_LOG("pull model output data failed");
                return FAILED;
            }

            std::string modelOutputTxtfileNameCur = modelOutputBinfileNameCur + ".txt";
            ret = Utils::SaveModelOutputData(modelOutputBinfileNameCur.c_str(), modelOutputTxtfileNameCur.c_str());
            if (ret != SUCCESS) {
                ERROR_LOG("save model output data failed");
                return FAILED;
            }
            processDvpp.DestroyOutputPara();
            processModel.DestroyInput();
        }
    }
    processDvpp.DestroyResource();
    rmdir(outFolder_);

    return SUCCESS;
}

void SampleProcess::DestroyResource()
{
    aclError ret;
    (void)aclrtUnSubscribeReport(static_cast<uint64_t>(threadId_), stream_);
    // destory thread
    runFlag = false;
    void *res = nullptr;
    int joinThreadErr = pthread_join(threadId_, &res);
    if (joinThreadErr != 0) {
        ERROR_LOG("join thread failed, threadId = %lu, err = %d", threadId_, joinThreadErr);
    } else {
        if ((uint64_t)res != 0) {
            ERROR_LOG("thread run failed. ret is %lu.", (uint64_t)res);
        }
    }
    INFO_LOG("destory thread success.");

    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy context failed");
        }
        context_ = nullptr;
    }
    INFO_LOG("end to destroy context");

    ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("reset device failed");
    }
    INFO_LOG("end to reset device is %d", deviceId_);

    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("finalize acl failed");
    }
    INFO_LOG("end to finalize acl");
}
