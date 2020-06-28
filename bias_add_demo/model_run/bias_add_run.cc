#include <iostream>
#include "acl/acl.h"
using namespace std;
bool g_isDevice = false;

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

Result InitResource()
{
   // ACL init
   const char *aclConfigPath = "./config/acl.json";
   aclError ret = aclInit(aclConfigPath);
   if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl init failed");
        return FAILED;
    }
    INFO_LOG("acl init success");
}

int main()
{
    //init resources
    Result ret = InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("sample init resource failed");
        return FAILED;
    }
    return SUCCESS;
}
