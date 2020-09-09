/**
* @file utils.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "utils.h"

bool RunStatus::isDevice_ = false;

bool Utils::ReadFileToDeviceMem(const char *fileName, void *&dataDev, uint32_t &dataSize)
{
    // read data from file.
    FILE *fp = fopen(fileName, "rb+");
    if (fp == nullptr) {
        ERROR_LOG("open file %s failed.\n", fileName);
        return false;
    }

    fseek(fp, 0, SEEK_END);
    long fileLenLong = ftell(fp);
    if (fileLenLong <= 0) {
        ERROR_LOG("file %s len is invalid.\n", fileName);
        fclose(fp);
        return false;
    }
    fseek(fp, 0, SEEK_SET);

    auto fileLen = static_cast<uint32_t>(fileLenLong);
    dataSize = fileLen;
    size_t readSize;
    // Malloc input device memory
    auto aclRet = acldvppMalloc(&dataDev, dataSize);
    if (aclRet != ACL_ERROR_NONE) {
        ERROR_LOG("acl malloc dvpp data failed, dataSize=%u, ret=%d.\n", dataSize, aclRet);
        fclose(fp);
        return false;
    }

    if (!(RunStatus::GetDeviceStatus())) {
        void *dataHost = malloc(fileLen);
        if (dataHost == nullptr) {
            ERROR_LOG("malloc host data buffer failed. fileLen=%u\n", fileLen);
            (void)acldvppFree(dataDev);
            fclose(fp);
            return false;
        }

        readSize = fread(dataHost, 1, fileLen, fp);
        if (readSize < fileLen) {
            ERROR_LOG("need read file %s %u bytes, but only %zu read.\n", fileName, fileLen, readSize);
            free(dataHost);
            (void)acldvppFree(dataDev);
            fclose(fp);
            return false;
        }

        // copy input to device memory
        aclRet = aclrtMemcpy(dataDev, dataSize, dataHost, fileLen, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_ERROR_NONE) {
            ERROR_LOG("acl memcpy data to dev failed, fileLen=%u, ret=%d.\n", fileLen, aclRet);
            free(dataHost);
            (void)acldvppFree(dataDev);
            dataDev = nullptr;
            fclose(fp);
            return false;
        }
        free(dataHost);
    } else {
        readSize = fread(dataDev, 1, fileLen, fp);
        if (readSize < fileLen) {
            ERROR_LOG("need read file %s %u bytes, but only %zu read.\n", fileName, fileLen, readSize);
            (void)acldvppFree(dataDev);
            fclose(fp);
            return false;
        }
    }

    fclose(fp);
    return true;
}

bool Utils::WriteToFile(const char *fileName, void *dataDev, uint32_t dataSize)
{
    if (dataDev == nullptr) {
        ERROR_LOG("dataDev is nullptr!");
        return false;
    }

    // copy output to host memory
    void *data = nullptr;
    aclError aclRet;
    if (!(RunStatus::GetDeviceStatus())) {
        data = malloc(dataSize);
        if (data == nullptr) {
            ERROR_LOG("malloc host data buffer failed. dataSize=%u\n", dataSize);
            return false;
        }
        aclRet = aclrtMemcpy(data, dataSize, dataDev, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_ERROR_NONE) {
            ERROR_LOG("acl memcpy data to host failed, dataSize=%u, ret=%d.\n", dataSize, aclRet);
            free(data);
            return false;
        }
    } else {
        data = dataDev;
    }

    FILE *outFileFp = fopen(fileName, "wb+");
    if (outFileFp == nullptr) {
        ERROR_LOG("fopen out file %s failed, error=%s.\n", fileName, strerror(errno));
        free(data);
        return false;
    }

    bool ret = true;
    size_t writeRet = fwrite(data, 1, dataSize, outFileFp);
    if (writeRet != dataSize) {
        ERROR_LOG("need write %u bytes to %s, but only write %zu bytes, error=%s.\n",
                      dataSize, fileName, writeRet, strerror(errno));
        ret = false;
    }

    if (!(RunStatus::GetDeviceStatus())) {
        free(data);
    }
    fflush(outFileFp);
    fclose(outFileFp);
    return ret;
}

Result Utils::PullModelOutputData(aclmdlDataset *modelOutput, const char *fileName)
{
    size_t outDatasetNum = aclmdlGetDatasetNumBuffers(modelOutput);
    if (outDatasetNum == 0) {
        ERROR_LOG("aclmdlGetDatasetNumBuffers from model output failed, outDatasetNum = 0");
    }
    for (size_t i = 0; i < outDatasetNum; ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(modelOutput, i);
        if (dataBuffer == nullptr) {
            ERROR_LOG("aclmdlGetDatasetBuffer from model output failed.");
        }

        void* data = aclGetDataBufferAddr(dataBuffer);
        if (data == nullptr) {
            ERROR_LOG("aclGetDataBufferAddr from dataBuffer failed.");
        }
        size_t bufferSize = aclGetDataBufferSize(dataBuffer);
        INFO_LOG("output[%zu] DataBuffer, buffer addr = %p, buffer size = %zu",
                i, data, bufferSize);

        void* dataPtr = nullptr;
        aclError ret;
        if (!(RunStatus::GetDeviceStatus())) {
            ret = aclrtMallocHost(&dataPtr, bufferSize);
            if (ret !=  ACL_ERROR_NONE) {
                ERROR_LOG("malloc host data buffer failed.");
                return FAILED;
            }
            ret = aclrtMemcpy(dataPtr, bufferSize, data, bufferSize, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_ERROR_NONE) {
                (void)aclrtFreeHost(dataPtr);
                ERROR_LOG("aclrtMemcpy device to host failed.");
            }
            INFO_LOG("memcopy output data from device to host buffer success.");
        } else {
            dataPtr = data;
        }

        uint32_t len = static_cast<uint32_t>(bufferSize);
        FILE *outputFile = fopen(fileName, "w+");
        if (outputFile) {
            fwrite(static_cast<char *>(dataPtr), len, sizeof(char), outputFile);
            fclose(outputFile);
            INFO_LOG("create output file success, filename=%s, size=%u", fileName, len);
        } else {
            ERROR_LOG("create output file %s failed, size=%u", fileName, len);
        }

        if (!(RunStatus::GetDeviceStatus())) {
            (void)aclrtFreeHost(dataPtr);
        }
    }
    return SUCCESS;
}

Result Utils::CheckFile(const char* fileName)
{
    int i = 0;
    INFO_LOG( "start check result file:%s", fileName);
    while (i < 10) {
        std::ifstream f (fileName);
        if(f.good()) {
            break;
        }
        sleep(1);
        INFO_LOG("check result, wait time [%ds]", i+1);
        i++;
    }
    if (10 == i) {
        INFO_LOG( "check result failed, timeout,expect file:%s", fileName);
        return FAILED;
    }
    INFO_LOG( "check result success, file exist");
    return SUCCESS;
}

Result Utils::CheckFolder(const char* foldName)
{
    INFO_LOG( "start check result fold:%s", foldName);
    if (access(foldName , 0) == -1) {
        int flag=mkdir(foldName , 0777);
        if (flag == 0)
        {
            INFO_LOG( "make successfully.");
        } else {
            INFO_LOG( "make errorly.");
            return FAILED;
        }
    }
    INFO_LOG( "check result success, fold exist");
    return SUCCESS;
}

Result Utils::SaveModelOutputData(const char* srcfileName, const char* dstfileName)
{
    Result ret = CheckFile(srcfileName);
    if (ret != SUCCESS) {
        ERROR_LOG("model output file not exist.");
        return FAILED;
    }
    FILE *model_output;
    model_output = fopen(srcfileName,"rb" );

    FILE *model_output_txt;
    model_output_txt = fopen(dstfileName, "w+");
    INFO_LOG( "reselut file: [%s]",dstfileName);

    int i = 0;
    float prop = 0.0;
    std::map<float, int, std::greater<float>> mp;
    std::map<float, int>::iterator ite = mp.begin();
    while (feof(model_output) == 0) {
        fread(&prop, sizeof(float), 1, model_output);
        mp.insert(ite, std::map<float, int>::value_type(prop, i));
        fprintf(model_output_txt, "%f,%d", prop, i);
        i++;
        ite++;
    }
    fclose(model_output);
    ite = mp.begin();
    float sum = 0.0;
    float max = ite->first;
    int classType = ite->second;
    for (i = 0 ; i < 5; i++) {
        sum+=ite->first;
        ite++;
    }
    fprintf(model_output_txt, "classType[%d], top1[%f], top5[%f]", classType, max, sum);
    fclose(model_output_txt);
    INFO_LOG( "result:classType[%d],top1[%f],top5[%f]", classType,max,sum);
    return SUCCESS;
}
std::vector<std::string> Utils::readDir(const char* folder)
{
    std::vector<std::string> fileList;
    struct dirent *dirp;
    DIR* dir = opendir(folder);
    while ((dirp = readdir(dir)) != nullptr) {
        if (dirp->d_type == DT_REG) {
            fileList.push_back(dirp->d_name);
        }
    }
    closedir(dir);
    return fileList;
}
