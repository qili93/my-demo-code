/**
* @file utils.h
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#pragma once
#include <iostream>
#include <unistd.h>
#include <dirent.h>
#include <fstream>
#include <cstring>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <map>
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

typedef struct PicDesc {
    std::string picName;
    int width;
    int height;
}PicDesc;

class RunStatus {
public:
    static void SetDeviceStatus(bool isDevice)
    {
        isDevice_ = isDevice;
    }
    static bool GetDeviceStatus()
    {
        return isDevice_;
    }
private:
    RunStatus() {}
    ~RunStatus() {}
    static bool isDevice_;
};

/**
 * Utils
 */
class Utils {
public:
    /**
    * @brief get width and height of video
    * @param [in] fileName: file name
    * @param [in] buffer of input data
    * @param [in] dataSize: size of data
    * @return success or fail
    */
    static bool WriteToFile(const char *fileName, void *dataDev, uint32_t dataSize);

    /**
    * @brief get width and height of video
    * @param [in] fileName: file name
    * @param [out] buffer of input data
    * @param [out] dataSize: size of data
    * @return success or fail
    */
    static bool ReadFileToDeviceMem(const char *fileName, void *&dataDev, uint32_t &dataSize);

    /**
    * @brief pull model output data to file
    * @param [in] modelOutput: model output dataset
    * @param [in] fileName: file name
    * @return result
    */
    static Result PullModelOutputData(aclmdlDataset *modelOutput, const char *fileName);

    /**
    * @brief save model output data to dst file
    * @param [in] srcfileName: src file name
    * @param [in] dstfileName: dst file name
    * @return result
    */
    static Result SaveModelOutputData(const char* srcfileName, const char* dstfileName);

    /**
    * @brief save dvpp output data
    * @param [in] fileName: file name
    * @param [in] devPtr: dvpp output data device addr
    * @param [in] dataSize: dvpp output data size
    * @return result
    */
    static Result SaveDvppOutputData(const char *fileName, const void *devPtr, uint32_t dataSize);

    /**
    * @brief check file if exist
    * @param [in] fileName: file to check
    * @return result
    */
    static Result CheckFile(const char* fileName);

    /**
    * @brief check fold, if not exist, create it
    * @param [in] fileName: fold to check
    * @return result
    */
    static Result CheckFolder(const char* foldName);

    /**
    * @brief read file of a dir
    * @param [in] fileName: folder
    * @return fileList
    */
    static std::vector<std::string> readDir(const char* folder);
};

