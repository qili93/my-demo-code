#!/bin/bash
ANDROID_ABI=armeabi-v7a # arm64-v8a
if [ "x$1" != "x" ]; then
    ANDROID_ABI=$1
fi

#rm ../../libs/PaddleLite/${ANDROID_ABI}/lib/libpaddle_light_api_shared.so
#cp /Work/Paddle/hongming/huawei/kirin/Paddle-Lite/build.lite.android.armv7.gcc/inference_lite_lib.android.armv7.npu/cxx/lib/libpaddle_full_api_shared.so ../../libs/PaddleLite/${ANDROID_ABI}/lib/
#rm ./libpaddle_light_api_shared.so
#curl -O http://yq01-gpu-255-129-13-00.epc.baidu.com:8213/Paddle/huawei/kirin/Paddle-Lite/build.lite.android.armv7.clang/inference_lite_lib.android.armv7/cxx/lib/libpaddle_light_api_shared.so
#cp ./libpaddle_light_api_shared.so ../../libs/PaddleLite/${ANDROID_ABI}/lib/

rm ../../libs/PaddleLite/${ANDROID_ABI}/lib/libpaddle_full_api_shared.so
#cp /Work/Paddle/hongming/huawei/kirin/Paddle-Lite/build.lite.android.armv7.gcc/inference_lite_lib.android.armv7.npu/cxx/lib/libpaddle_full_api_shared.so ../../libs/PaddleLite/${ANDROID_ABI}/lib/
rm ./libpaddle_full_api_shared.so
curl -O http://yq01-gpu-255-129-13-00.epc.baidu.com:8214/Paddle/huawei/kirin/Paddle-Lite/build.lite.android.armv7.clang/inference_lite_lib.android.armv7/cxx/lib/libpaddle_full_api_shared.so
cp ./libpaddle_full_api_shared.so ../../libs/PaddleLite/${ANDROID_ABI}/lib/
rm ./libpaddle_full_api_shared.so
rm ./libpaddle_light_api_shared.so
