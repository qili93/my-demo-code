/**
* @file vdec_process.h
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#pragma once
#include <cstdint>
#include <iostream>
#include "utils.h"
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

/**
 * VdecProcess
 */
class VdecProcess {
public:
    /**
    * @brief Constructor
    */
    VdecProcess();

    /**
    * @brief Destructor
    */
    ~VdecProcess();

    /**
    * @brief vdec global init
    * @param [in] threadId: index of thread
    * @param [in] enType: type of input stream
    * @param [in] format: format of pic
    * @return result
    */
    Result InitResource(pthread_t threadId, int32_t enType, int32_t format);

    /**
    * @brief set vdec input
    * @param [in] inBufferDev: input buffer
    * @param [in] inBufferSize: buffer size
    * @param [in] inputWidth:width of pic
    * @param [in] inputHeight:height of pic
    */
    void SetInput(void *inBufferDev, uint32_t inBufferSize, int inputWidth, int inputHeight);

    /**
    * @brief destroy StreamDesc
    */
    void DestroyStreamDesc();

    /**
    * @brief destroy PicDesc
    */
    void DestroyPicDesc();

    /**
    * @brief destroy resource
    */
    void DestroyResource();

    /**
    * @brief vdec process
    * @return result
    */
    Result Process();

private:
    Result CreateStreamDesc();
    Result CreatePicDesc(size_t size);

    pthread_t threadId_;

    aclvdecChannelDesc *vdecChannelDesc_;
    acldvppStreamDesc *streamInputDesc_;
    acldvppPicDesc *picOutputDesc_;
    void *picOutBufferDev_;
    void *inBufferDev_;
    uint32_t inBufferSize_;
    uint32_t inputWidth_;
    uint32_t inputHeight_;
    int32_t format_;
    int32_t enType_;
};

