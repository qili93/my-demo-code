/**
* @file vdec_process.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "vdec_process.h"

using namespace std;

VdecProcess::VdecProcess()
    : vdecChannelDesc_(nullptr), streamInputDesc_(nullptr),
      picOutputDesc_(nullptr), picOutBufferDev_(nullptr),
      inBufferDev_(nullptr), inBufferSize_(0), inputWidth_(0),
      inputHeight_(0), format_(0), enType_(0)
{
}

VdecProcess::~VdecProcess()
{
}

void callback(acldvppStreamDesc *input, acldvppPicDesc *output, void *userdata)
{
    static int count = 1;
    if (output != nullptr) {
        void *vdecOutBufferDev = acldvppGetPicDescData(output);
        if (vdecOutBufferDev != nullptr) {
            // 0: vdec success; others, vdec failed
            int retCode = acldvppGetPicDescRetCode(output);
            if (retCode == 0) {
                // process task: write file
                uint32_t size = acldvppGetPicDescSize(output);
                std::string fileNameSave = "outdir/image" + std::to_string(count);
                if (!Utils::WriteToFile(fileNameSave.c_str(), vdecOutBufferDev, size)) {
                    ERROR_LOG("write file failed.");
                }
            } else {
                ERROR_LOG("vdec decode frame failed.");
            }

            // free output vdecOutBufferDev
            aclError ret = acldvppFree(vdecOutBufferDev);
            if (ret != ACL_ERROR_NONE) {
                ERROR_LOG("fail to free output pic desc data ret = %d", ret);
            }
        }
        // destroy pic desc
        aclError ret = acldvppDestroyPicDesc(output);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("fail to destroy output pic desc");
        }
    }

    // free input vdecInBufferDev and destroy stream desc
    if (input != nullptr) {
        void *vdecInBufferDev = acldvppGetStreamDescData(input);
        if (vdecInBufferDev != nullptr) {
            aclError ret = acldvppFree(vdecInBufferDev);
            if (ret != ACL_ERROR_NONE) {
                ERROR_LOG("fail to free input stream desc data");
            }
        }
        aclError ret = acldvppDestroyStreamDesc(input);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("fail to destroy input stream desc");
        }
    }

    INFO_LOG("success to callback %d.", count);
    count++;
}

Result VdecProcess::InitResource(pthread_t threadId, int32_t enType, int32_t format)
{
    threadId_ = threadId;
    format_ = format;
    enType_ = enType;
    // create vdec channelDesc
    vdecChannelDesc_ = aclvdecCreateChannelDesc();
    if (vdecChannelDesc_ == nullptr) {
        ERROR_LOG("fail to create vdec channel desc");
        return FAILED;
    }

    // channelId: 0-15
    aclError ret = aclvdecSetChannelDescChannelId(vdecChannelDesc_, 10);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("fail to set vdec ChannelId");
        return FAILED;
    }

    ret = aclvdecSetChannelDescThreadId(vdecChannelDesc_, threadId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("fail to create threadId");
        return FAILED;
    }

    // callback func
    ret = aclvdecSetChannelDescCallback(vdecChannelDesc_, callback);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("fail to set vdec Callback");
        return FAILED;
    }

    ret = aclvdecSetChannelDescEnType(vdecChannelDesc_, static_cast<acldvppStreamFormat>(enType_));
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("fail to set vdec EnType");
        return FAILED;
    }

    ret = aclvdecSetChannelDescOutPicFormat(vdecChannelDesc_, static_cast<acldvppPixelFormat>(format_));
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("fail to set vdec OutPicFormat");
        return FAILED;
    }

    // create vdec channel
    ret = aclvdecCreateChannel(vdecChannelDesc_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("fail to create vdec channel");
        return FAILED;
    }

    INFO_LOG("vdec init resource success");
    return SUCCESS;
}

void VdecProcess::SetInput(void *inBufferDev, uint32_t inBufferSize,
                          int inputWidth, int inputHeight)
{
    inBufferDev_ = inBufferDev;
    inBufferSize_ = inBufferSize;
    inputWidth_ = inputWidth;
    inputHeight_ = inputHeight;
}

Result VdecProcess::CreateStreamDesc()
{
    // create input stream desc
    streamInputDesc_ = acldvppCreateStreamDesc();
    if (streamInputDesc_ == nullptr) {
        ERROR_LOG("fail to create input stream desc");
        return FAILED;
    }

    aclError ret = acldvppSetStreamDescData(streamInputDesc_, inBufferDev_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("fail to set data for stream desc");
        return FAILED;
    }
    // set size for dvpp stream desc
    ret = acldvppSetStreamDescSize(streamInputDesc_, inBufferSize_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("fail to set size for stream desc");
        return FAILED;
    }
    return SUCCESS;
}

void VdecProcess::DestroyStreamDesc()
{
    if (inBufferDev_ != nullptr) {
        (void)acldvppFree(inBufferDev_);
        inBufferDev_ = nullptr;
    }
    if (streamInputDesc_ != nullptr) {
        (void)acldvppDestroyStreamDesc(streamInputDesc_);
        streamInputDesc_ = nullptr;
    }
}

Result VdecProcess::CreatePicDesc(size_t size)
{
    // Malloc output device memory
    aclError ret = acldvppMalloc(&picOutBufferDev_, size);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclrtMalloc failed, ret=%d", ret);
        return FAILED;
    }
    picOutputDesc_ = acldvppCreatePicDesc();
    if (picOutputDesc_ == nullptr) {
        ERROR_LOG("fail to create output pic desc");
        return FAILED;
    }
    ret = acldvppSetPicDescData(picOutputDesc_, picOutBufferDev_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("fail to set PicDescData");
        return FAILED;
    }
    ret = acldvppSetPicDescSize(picOutputDesc_, size);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("fail to set PicDescSize");
        return FAILED;
    }
    ret = acldvppSetPicDescFormat(picOutputDesc_, static_cast<acldvppPixelFormat>(format_));
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("fail to set PicDescHeight");
        return FAILED;
    }
    return SUCCESS;
}

void VdecProcess::DestroyPicDesc()
{
    if (picOutBufferDev_ != nullptr) {
        (void)acldvppFree(picOutBufferDev_);
        picOutBufferDev_ = nullptr;
    }
    if (picOutputDesc_ != nullptr) {
        (void)acldvppDestroyPicDesc(picOutputDesc_);
        picOutputDesc_ = nullptr;
    }
}

Result VdecProcess::Process()
{
     // create stream desc
    Result err = CreateStreamDesc();
    if (err != SUCCESS) {
        DestroyStreamDesc();
        return FAILED;
    }
    // create pic desc
    size_t DataSize = (inputWidth_ * inputHeight_ * 3) / 2;
    err = CreatePicDesc(DataSize);
    if (err != SUCCESS) {
        DestroyStreamDesc();
        DestroyPicDesc();
        return FAILED;
    }
    // send frame
    aclError ret = aclvdecSendFrame(vdecChannelDesc_, streamInputDesc_,
                                    picOutputDesc_, nullptr, nullptr);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("fail to send frame, ret=%u", ret);
        DestroyStreamDesc();
        DestroyPicDesc();
        return FAILED;
    }
    return SUCCESS;
}

void VdecProcess::DestroyResource()
{
    if (vdecChannelDesc_ != nullptr) {
        aclError ret = aclvdecDestroyChannel(vdecChannelDesc_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("acldvppDestroyChannel failed, ret = %d", ret);
        }
        aclvdecDestroyChannelDesc(vdecChannelDesc_);
        vdecChannelDesc_ = nullptr;
    }
}
