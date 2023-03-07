import os


try:
    """Git clone PaddleClas repo"""
 
    os.system(f"git clone https://openi.pcl.ac.cn/PaddlePaddle/PaddleClas.git -b develop")

    print("Succeed to clone paddleclas repo")
except:
    print("Failed to git clone PaddleClas")


try:
    """Run prepare for PaddleClas TIPC"""
 
    os.system(f"cd PaddleClas && bash test_tipc/prepare.sh test_tipc/configs/ResNet/ResNet50_train_infer_python.txt 'lite_train_lite_infer'")

    print("Succeed to prepare data for paddleclas tipc")
except:
    print("Failed to prepare data for paddleclas tipc")


try:
    """Run tipc for PaddleClas"""

    os.system("cd PaddleClas && bash test_tipc/test_train_inference_python_npu.sh test_tipc/configs/ResNet/ResNet50_train_infer_python.txt 'lite_train_lite_infer'")
 
    print("Succeed to run tipc paddleclas resnet50")
except:
    print("Failed to run tipc paddleclas resnet50")


try:
    """Get output of TIPC result"""

    os.system("cd PaddleClas && cat test_tipc/output/ResNet50/lite_train_lite_infer/results_python.log")
 
    print("Succeed to get output of TIPC result")
except:
    print("Failed to get output of TIPC result")
