#include "graph/graph.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "graph/ge_error_codes.h"
#include "ge/ge_api_types.h"
#include "ge/ge_ir_build.h"
#include "all_ops.h" // opp/op_proto/built-in/inc

#include "model_build.h"

bool OMModelBuild::GenGraph(ge::Graph& graph) {
    // input data op => format: NCHW
    // x: A 4D tensor of input images.
    ge::TensorDesc input_desc(ge::Shape({ 1, 1, 4, 4 }), ge::FORMAT_ND, ge::DT_FLOAT);
    auto input_x = ge::op::Data("input_x");
    input_x.update_input_desc_x(input_desc);
    input_x.update_output_desc_y(input_desc);

    // filter tensor => format: HWCN
    // filter: A 4D tensor of filters.
    auto filter_shape = ge::Shape({ 2, 2, 1, 1 });
    ge::TensorDesc filter_desc(filter_shape, ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor filter_tensor(filter_desc);
    int64_t filter_data_size = filter_shape.GetShapeSize();
    int64_t filter_data_length = filter_data_size * sizeof(float);
    // generating random data to filter tensor between 0 to 1
    srand (static_cast <unsigned> (time(0)));
    float * filter_data = new(std::nothrow) float[filter_data_size];
    for (int64_t j = 0; j < filter_data_size; j++) {
      filter_data[j] = static_cast <float> (rand()) /( static_cast <float> (RAND_MAX));
    }
    auto status = filter_tensor.SetData(reinterpret_cast<uint8_t*>(filter_data), filter_data_length);
    if (status != ge::GRAPH_SUCCESS) {
      LOG(ERROR) << "Set Filter Tensor Data Failed";
      delete [] filter_data;
      return false;
    }
    // const op of filter
    auto conv_filter = ge::op::Const("Conv2D/filter").set_attr_value(filter_tensor);

    // bias tensor => bias: An optional 1D tensor.
    auto bias_shape = ge::Shape({1});
    ge::TensorDesc bias_desc(bias_shape, ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor bias_tensor(bias_desc);
    int64_t bias_data_size = bias_shape.GetShapeSize();
    int64_t bias_data_length = bias_data_size * sizeof(float);
    // generating random data to bias tensor between 0 to 1
    float * bias_data = new(std::nothrow) float[bias_data_size];
    for (int64_t j = 0; j < bias_data_size; j++) {
      bias_data[j] = static_cast <float> (rand()) /( static_cast <float> (RAND_MAX));
    }
    status = bias_tensor.SetData(reinterpret_cast<uint8_t*>(bias_data), bias_data_length);
    if (status != ge::GRAPH_SUCCESS) {
      LOG(ERROR) << "Set Bias Tensor Data Failed";
      delete [] bias_data;
      return false;
    }
    // const op of bias
    auto conv_bias = ge::op::Const("Conv2D/bias").set_attr_value(bias_tensor);

    // conv op
    auto conv_op = ge::op::Conv2D("conv1");
    conv_op.set_input_x(input_x);
    conv_op.set_input_filter(conv_filter);
    conv_op.set_input_bias(conv_bias);
    conv_op.set_attr_strides({ 1, 1, 1, 1 }); // same format to input X
    conv_op.set_attr_pads({ 0, 0, 0, 0 }); //top, bottom, left and right
    conv_op.set_attr_dilations({ 1, 1, 1, 1 }); // same format to input X
    conv_op.set_attr_groups(1); //must set to 1
    conv_op.set_attr_data_format("NCHW"); // string

    ge::TensorDesc conv2d_input_desc_x(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorDesc conv2d_input_desc_filter(ge::Shape(), ge::FORMAT_HWCN, ge::DT_FLOAT);
    ge::TensorDesc conv2d_output_desc_y(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
    conv_op.update_input_desc_x(conv2d_input_desc_x);
    conv_op.update_input_desc_filter(conv2d_input_desc_filter);
    conv_op.update_output_desc_y(conv2d_output_desc_y);

    auto relu1 = ge::op::Relu("relu");
    relu1.set_input_x(conv_op, "y");

    // Build Graph
    std::vector<ge::Operator> inputs{ input_x };
    std::vector<ge::Operator> outputs{ relu1 };
    std::vector<std::pair<ge::Operator, std::string>> outputs_with_name = {{relu1, "y"}};

    graph.SetInputs(inputs).SetOutputs(outputs);
    return true;
}

bool OMModelBuild::SaveModel(ge::Graph& om_graph, std::string model_path)
{
    LOG(INFO) << "-------Enter: [model_build](SaveModel)-------";
    // 1. Genetate graph
    // ge::Graph om_graph("bias_add_graph");
    // if(!GenGraph(om_graph)) {
    //   LOG(ERROR) << "Generate BiasAdd Graph Failed!");
    // }
    // LOG(INFO) << "Generate BiasAdd Graph SUCCESS!");

    // 2. system init
    std::map<std::string, std::string> global_options = {
        {ge::ir_option::SOC_VERSION, "Ascend310"},
    };
    if (ge::aclgrphBuildInitialize(global_options) !=  ge::GRAPH_SUCCESS) {
      LOG(ERROR) << "[model_build](SaveModel) aclgrphBuildInitialize Failed!";
    } else {
      LOG(INFO) << "[model_build](SaveModel) aclgrphBuildInitialize succees";
    }

    // 3. Build IR Model
    ge::ModelBufferData model_om_buffer;
    std::map<std::string, std::string> options;
    //PrepareOptions(options);

    if (ge::aclgrphBuildModel(om_graph, options, model_om_buffer) !=  ge::GRAPH_SUCCESS) {
      LOG(ERROR) << "[model_build](SaveModel) aclgrphBuildModel Failed!";
    } else {
        LOG(INFO) << "[model_build](SaveModel) aclgrphBuildModel succees";
    }

    // 4. Save IR Model
    if (ge::aclgrphSaveModel(model_path, model_om_buffer) != ge::GRAPH_SUCCESS) {
      LOG(ERROR) << "[model_build](SaveModel) aclgrphSaveModel Failed!";
    } else {
        LOG(INFO) << "[model_build](SaveModel) aclgrphSaveModel succees";
    }

    // 5. release resource
    ge::aclgrphBuildFinalize();
    LOG(INFO) << "-------Leave: [model_build](SaveModel)-------";
    return true;
}