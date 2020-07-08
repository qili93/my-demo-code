/**
* @file main.cpp
*
* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string.h>
#include "graph/graph.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "graph/ge_error_codes.h"
#include "ge/ge_api_types.h"
#include "ge/ge_ir_build.h"
#include "all_ops.h"
#include <dlfcn.h>
#include <unistd.h>
//#include "add.h" // custom op ,if you have one new or different op defination with frame's,please
                   // add head file here.If same with frame , no need to add head file here

using namespace std;
using namespace ge;
using ge::Operator;

static const std::string PATH = "./data/";

void PrepareOptions(std::map<std::string, std::string>& options) {
}

bool GetConstTensorFromBin(string path, Tensor &weight, uint32_t len) {
    ifstream in_file(path.c_str(), std::ios::in | std::ios::binary);
    if (!in_file.is_open()) {
        std::cout << "failed to open" << path.c_str() << '\n';
        return false;
    }
    in_file.seekg(0, ios_base::end);
    istream::pos_type file_size = in_file.tellg();
    in_file.seekg(0, ios_base::beg);

    if (len != file_size) {
        cout << "Invalid Param.len:" << len << " is not equal with binary size（" << file_size << ")\n";
        in_file.close();
        return false;
    }
    char* pdata = new(std::nothrow) char[len];
    if (pdata == nullptr) {
        cout << "Invalid Param.len:" << len << " is not equal with binary size（" << file_size << ")\n";
        in_file.close();
        return false;
    }
    in_file.read(reinterpret_cast<char*>(pdata), len);
    auto status = weight.SetData(reinterpret_cast<uint8_t*>(pdata), len);
    if (status != ge::GRAPH_SUCCESS) {
        cout << "Set Tensor Data Failed"<< "\n";
        delete [] pdata;
        in_file.close();
        return false;
    }
    in_file.close();
    return true;
}
bool GenGraph(Graph& graph)
{
    auto shape_data = vector<int64_t>({ 1,1,28,28 });
    TensorDesc desc_data(ge::Shape(shape_data), FORMAT_ND, DT_FLOAT);

    // data op
    auto data = op::Data("data");
    data.update_input_desc_x(desc_data);
    data.update_output_desc_y(desc_data);
    // custom op ,using method is the same with frame internal op
    // [Notice]: if you want to use custom self-define op, please prepare custom op according to custum op define user guides
    auto add = op::Add("add")
        .set_input_x1(data)
        .set_input_x2(data);
    // AscendQuant
    auto quant = op::AscendQuant("quant")
        .set_input_x(data)
        .set_attr_scale(1.0)
        .set_attr_offset(0.0);

    // const op: conv2d weight
    auto weight_shape = ge::Shape({ 2,2,1,1 });
    TensorDesc desc_weight_1(weight_shape, FORMAT_ND, DT_INT8);
    Tensor weight_tensor(desc_weight_1);
    uint32_t weight_1_len = weight_shape.GetShapeSize();
    bool res = GetConstTensorFromBin(PATH+"Conv2D_kernel_quant.bin", weight_tensor, weight_1_len);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
        return -1;
    }
    auto conv_weight = op::Const("Conv2D/weight")
        .set_attr_value(weight_tensor);

    // conv2d op
    auto conv2d = op::Conv2D("Conv2d1")
        .set_input_x(quant)
        .set_input_filter(conv_weight)
        .set_attr_strides({ 1, 1, 1, 1 })
        .set_attr_pads({ 0, 1, 0, 1 })
        .set_attr_dilations({ 1, 1, 1, 1 });

    TensorDesc conv2d_input_desc_x(ge::Shape(), FORMAT_NCHW, DT_INT8);
    TensorDesc conv2d_input_desc_filter(ge::Shape(), FORMAT_HWCN, DT_INT8);
    TensorDesc conv2d_output_desc_y(ge::Shape(), FORMAT_NCHW, DT_INT8);
    conv2d.update_input_desc_x(conv2d_input_desc_x);
    conv2d.update_input_desc_filter(conv2d_input_desc_filter);
    conv2d.update_output_desc_y(conv2d_output_desc_y);
    // dequant scale
    TensorDesc desc_dequant_shape(ge::Shape({ 1 }), FORMAT_ND, DT_UINT64);
    Tensor dequant_tensor(desc_dequant_shape);
    uint64_t dequant_scale_val = 1;
    auto status = dequant_tensor.SetData(reinterpret_cast<uint8_t*>(&dequant_scale_val), sizeof(uint64_t));
    if (status != ge::GRAPH_SUCCESS) {
        cout << __LINE__ << "Set Tensor Data Failed" << "\n";
        return false;
    }
    auto dequant_scale = op::Const("dequant_scale")
        .set_attr_value(dequant_tensor);

    // AscendDequant
    auto dequant = op::AscendDequant("dequant")
        .set_input_x(conv2d)
        .set_input_deq_scale(dequant_scale);

    // const op: BiasAdd weight
    auto weight_bias_add_shape_1 = ge::Shape({ 1 });
    TensorDesc desc_weight_bias_add_1(weight_bias_add_shape_1, FORMAT_ND, DT_FLOAT);
    Tensor weight_bias_add_tensor_1(desc_weight_bias_add_1);
    uint32_t weight_bias_add_len_1 = weight_bias_add_shape_1.GetShapeSize() * sizeof(float);
    float weight_bias_add_value = 0.006448820233345032;
    status = weight_bias_add_tensor_1.SetData(reinterpret_cast<uint8_t*>(&weight_bias_add_value), weight_bias_add_len_1);
    if (status != ge::GRAPH_SUCCESS) {
        cout << __LINE__ << "Set Tensor Data Failed" << "\n";
        return false;
    }
    auto bias_weight_1 = op::Const("Bias/weight_1")
        .set_attr_value(weight_bias_add_tensor_1);
    // BiasAdd 1
    auto bias_add_1 = op::BiasAdd("bias_add_1")
        .set_input_x(dequant)
        .set_input_bias(bias_weight_1)
        .set_attr_data_format("NCHW");

    // const
    int32_t value[2] = {1,-1};

    auto value_shape = ge::Shape({ 2 });
    TensorDesc desc_dynamic_const(value_shape, FORMAT_ND, DT_INT32);
    Tensor dynamic_const_tensor(desc_dynamic_const);
    uint32_t dynamic_const_len = value_shape.GetShapeSize() * sizeof(int32_t);
    status = dynamic_const_tensor.SetData(reinterpret_cast<uint8_t*>(&(value[0])), dynamic_const_len);
    if (status != ge::GRAPH_SUCCESS) {
        cout << __LINE__ << "Set Tensor Data Failed" << "\n";
        return false;
    }
    auto dynamic_const = op::Const("dynamic_const").set_attr_value(dynamic_const_tensor);

    // ReShape op
    auto reshape = op::Reshape("Reshape")
        .set_input_x(bias_add_1)
        .set_input_shape(dynamic_const);
    // MatMul + BiasAdd
    // MatMul weight 1
    auto matmul_weight_shape_1 = ge::Shape({784,512});
    TensorDesc desc_matmul_weight_1(matmul_weight_shape_1, FORMAT_ND, DT_FLOAT);
    Tensor matmul_weight_tensor_1(desc_matmul_weight_1);
    uint32_t matmul_weight_1_len = matmul_weight_shape_1.GetShapeSize() * sizeof(float);
    res = GetConstTensorFromBin(PATH + "dense_kernel.bin", matmul_weight_tensor_1, matmul_weight_1_len);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
        return -1;
    }
    auto matmul_weight_1 = op::Const("dense/kernel")
        .set_attr_value(matmul_weight_tensor_1);
    // MatMul1
    auto matmul_1 = op::MatMul("MatMul_1")
        .set_input_x1(reshape)
        .set_input_x2(matmul_weight_1);
    // BiasAdd const 2
    auto bias_add_shape_2 = ge::Shape({ 512 });
    TensorDesc desc_bias_add_const_1(bias_add_shape_2, FORMAT_ND, DT_FLOAT);
    Tensor bias_add_const_tensor_1(desc_bias_add_const_1);
    uint32_t bias_add_const_len_1 = bias_add_shape_2.GetShapeSize() * sizeof(float);
    res = GetConstTensorFromBin(PATH + "dense_bias.bin", bias_add_const_tensor_1, bias_add_const_len_1);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
        return -1;
    }
    auto bias_add_const_1 = op::Const("dense/bias")
        .set_attr_value(bias_add_const_tensor_1);
    // BiasAdd 2
    auto bias_add_2 = op::BiasAdd("bias_add_2")
        .set_input_x(matmul_1)
        .set_input_bias(bias_add_const_1)
        .set_attr_data_format("NCHW");
    // Relu6
    auto relu6 = op::Relu6("relu6")
        .set_input_x(bias_add_2);
    // MatMul weight 2
    auto matmul_weight_shape_2 = ge::Shape({ 512, 10 });
    TensorDesc desc_matmul_weight_2(matmul_weight_shape_2, FORMAT_ND, DT_FLOAT);
    Tensor matmul_weight_tensor_2(desc_matmul_weight_2);
    uint32_t matmul_weight_2_len = matmul_weight_shape_2.GetShapeSize() * sizeof(float);
    res = GetConstTensorFromBin(PATH + "OutputLayer_kernel.bin", matmul_weight_tensor_2, matmul_weight_2_len);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
        return -1;
    }
    auto matmul_weight_2 = op::Const("OutputLayer/kernel")
        .set_attr_value(matmul_weight_tensor_2);
    // MatMul 2
    auto matmul_2 = op::MatMul("MatMul_2")
        .set_input_x1(relu6)
        .set_input_x2(matmul_weight_2);
    // BiasAdd const 3
    auto bias_add_shape_3 = ge::Shape({ 10 });
    TensorDesc desc_bias_add_const_3(bias_add_shape_3, FORMAT_ND, DT_FLOAT);
    Tensor bias_add_const_tensor_3(desc_bias_add_const_3);
    uint32_t bias_add_const_len_3 = bias_add_shape_3.GetShapeSize() * sizeof(float);
    res = GetConstTensorFromBin(PATH + "OutputLayer_bias.bin", bias_add_const_tensor_3, bias_add_const_len_3);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
        return -1;
    }
    auto bias_add_const_3 = op::Const("OutputLayer/bias")
        .set_attr_value(bias_add_const_tensor_3);
    // BiasAdd 3
    /*
     * When set input for some node, there are two methodes for you.
     * Method 1: operator level method. Frame will auto connect the node's output edge to netoutput nodes for user
     *   we recommend this method when some node own only one out node
     * Method 2: edge of operator level. Frame will find the edge according to the output edge name
     *   we recommend this method when some node own multi out nodes and only one out edge data wanted back
     */
    auto bias_add_3 = op::BiasAdd("bias_add_3")
        .set_input_x(matmul_2, "y")
        .set_input_bias(bias_add_const_3, "y")
        .set_attr_data_format("NCHW");
    // Softmax op
    auto softmax = op::SoftmaxV2("Softmax")
        .set_input_x(bias_add_3, "y");

    std::vector<Operator> inputs{ data };
    /*
     * The same as set input, when point net output ,Davince framework alos support multi method to set outputs info
     * Method 1: operator level method. Frame will auto connect the node's output edge to netoutput nodes for user
     *   we recommend this method when some node own only one out node
     * Method 2: edge of operator level. Frame will find the edge according to the output edge name
     *   we recommend this method when some node own multi out nodes and only one out edge data wanted back
     * Using method is like follows:
     */
    std::vector<Operator> outputs{ softmax, add };
    std::vector<std::pair<ge::Operator, std::string>> outputs_with_name = {{softmax, "y"}};

    graph.SetInputs(inputs).SetOutputs(outputs);

    return true;
}

int main(int argc, char* argv[])
{
    cout << "========== Test Start ==========" << endl;
    if (argc != 2) {
        cout << "[ERROR]input arg num must be 2! " << endl;
        cout << "The second arg stand for soc version! Please retry with your soc version " << endl;
        cout << "[Notice] Supported soc version as list:Ascend310 Ascend910 Ascend610 Ascend620 Hi3796V300ES" << endl;
        return -1;
    }
    cout << argv[1] << endl;

    // 1. Genetate graph
    Graph graph1("IrGraph1");
    bool ret = GenGraph(graph1);

    if (!ret) {
        cout << "========== Generate Graph1 Failed! ==========" << endl;
        return -1;
    }
    else {
        cout << "========== Generate Graph1 Success! ==========" << endl;
    }

    // 2. system init
    std::map<std::string, std::string> global_options = {
        {ge::ir_option::SOC_VERSION, string(argv[1])},
        {"ge.quantOptimize", "true"},
    };
    auto status = aclgrphBuildInitialize(global_options);
    // 3. Build Ir Model1
    ModelBufferData model1;
    std::map<std::string, std::string> options;
    PrepareOptions(options);

    status = aclgrphBuildModel(graph1, options, model1);
    if (status == GRAPH_SUCCESS) {
        cout << "Build Model1 SUCCESS!" << endl;
    }
    else {
        cout << "Build Model1 Failed!" << endl;
    }
    // 4. Save Ir Model
    status = aclgrphSaveModel("resnet50_build", model1);
    if (status == GRAPH_SUCCESS) {
        cout << "Save Offline Model1 SUCCESS!" << endl;
    }
    else {
        cout << "Save Offline Model1 Failed!" << endl;
    }

    // release resource
    aclgrphBuildFinalize();
    return 0;
}
