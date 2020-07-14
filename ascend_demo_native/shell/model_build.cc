#include "graph/graph.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "graph/ge_error_codes.h"
#include "ge/ge_api_types.h"
#include "ge/ge_ir_build.h"
#include "all_ops.h" // opp/op_proto/built-in/inc

#include "model_build.h"

bool OMModelBuild::GenGraph(ge::Graph& graph) {
    // ==========================input data op => format: NCHW==========================
    // x: A 4D tensor of input images.
    ge::TensorDesc input_desc(ge::Shape({ 1, 1, 28, 28 }), ge::FORMAT_ND, ge::DT_FLOAT);
    auto input_x = ge::op::Data("input_x");
    input_x.update_input_desc_x(input_desc);
    input_x.update_output_desc_y(input_desc);

    DebugGeOPInfo("Input OP", &input_x);

    // ==========================filter const op => format: NCHW==========================
    // filter: A 4D tensor of filters.
    auto filter_shape = ge::Shape({ 4, 1, 5, 5 }); // HCHW = oc, ic, h, w
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

    DebugGeOPInfo("Filter OP", &conv_filter);

    // ==========================Bias Const OP => format ND {oc}==========================
    // bias: An optional 1D tensor. => {oc}
    auto bias_shape = ge::Shape({4});
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

    DebugGeOPInfo("Bias OP", &conv_bias);

    // ========================== Conv2D OP ==========================
    // Output: y: A 4D Tensor of output images.
    auto conv_op = ge::op::Conv2D("conv1");
    // Input: x: A 4D tensor of input images.
    conv_op.set_input_x(input_x); // {1, 1, 4, 4}
    // filter: A 4D tensor of filters - H, W, C = ic, N = oc
    conv_op.set_input_filter(conv_filter);
    // bias: 1D tensor of foramt ND
    conv_op.set_input_bias(conv_bias);
    // filter: A 4D tensor of filters - H, W, C = ic, N = bs
    conv_op.set_attr_strides({ 1, 1, 1, 1 });
    // pads: A list of 4 integers. Specifying the top, bottom, left and right
    conv_op.set_attr_pads({ 0, 0, 0, 0 });
    // dilations: A list of 4 integers. same dimension order and value as strides
    conv_op.set_attr_dilations({ 1, 1, 1, 1 });
    // "groups". Must be set to 1.
    conv_op.set_attr_groups(1);
    // string from: "NHWC", "NCHW"
    conv_op.set_attr_data_format("NCHW");
    // update conv op input and output
    ge::TensorDesc conv2d_input_desc_x(input_desc.GetShape(), ge::FORMAT_NCHW, input_desc.GetDataType());
    ge::TensorDesc conv2d_input_desc_filter(filter_desc.GetShape(), ge::FORMAT_NCHW, filter_desc.GetDataType());
    ge::TensorDesc conv2d_input_desc_bias(bias_desc.GetShape(), ge::FORMAT_ND, bias_desc.GetDataType());
    ge::TensorDesc conv2d_output_desc_y(ge::Shape(), ge::FORMAT_NCHW, input_desc.GetDataType());
    conv_op.update_input_desc_x(conv2d_input_desc_x);
    conv_op.update_input_desc_filter(conv2d_input_desc_filter);
    conv_op.update_input_desc_bias(conv2d_input_desc_bias);
    conv_op.update_output_desc_y(conv2d_output_desc_y);

    DebugGeOPInfo("Conv OP 1", &conv_op);

    // //DepthwiseConv2D
    // auto deepwisec_conv_op = ge::op::DepthwiseConv2D("conv2");
    // // x: A 4D tensor of type float16, with shape [N, C, H, W] or [N, H, W, C]
    // deepwisec_conv_op.set_input_x(conv_op, "conv2_x"); // {1, 1, 3, 3}
    // // filter: A 4D tensor of type float16, with shape [H, W, C, K]
    // // The filter is 4D with shape [Hf, Wf, C, K], but the data is 6D with shape
    // // [C1, Hf, Wf, K, Co, C0], where K is fixed at 1, and Co and C0 are 16.
    // deepwisec_conv_op.set_input_filter(conv_filter, "conv2_filter");
    // // bias: 1D tensor of foramt ND, type float16 or int32
    // deepwisec_conv_op.set_input_bias(conv_bias, "conv2_bias");
    // // filter: A 4D tensor of filters
    // // Must be with shape [1, 1, stride_height, stride_width] or [1, stride_height, stride_width, 1].
    // deepwisec_conv_op.set_attr_strides({ 1, 1, 1, 1 });
    // // dilations: A list of 4 integers. 
    // // Must be with shape [1, 1, dilation_height, dilation_width] or [1, dilation_height, dilation_width, 1].
    // deepwisec_conv_op.set_attr_dilations({ 1, 1, 1, 1 });
    // // pads: pads: A required list or tuple. Padding added to each dimension of the input.
    // deepwisec_conv_op.set_attr_pads({ 0, 0, 0, 0 });
    // // "groups" NOT supported for DepthwiseConv2D
    // // deepwisec_conv_op.set_attr_groups(1);
    // // string from: "NHWC", "NCHW". Defaults to "NHWC".
    // deepwisec_conv_op.set_attr_data_format("NCHW");

    // deepwisec_conv_op.update_input_desc_x(conv2d_input_desc_x);
    // deepwisec_conv_op.update_input_desc_filter(conv2d_input_desc_filter);
    // deepwisec_conv_op.update_output_desc_y(conv2d_output_desc_y);

    // ========================== RELU OP ==========================
    auto relu1 = ge::op::Relu("relu");
    relu1.set_input_x(conv_op);

    // ========================== POOL OP ==========================
    // Ouput: y: An NCHW tensor of type float16, float32, int32.
    auto pool_op = ge::op::Pooling("pool1");
    // x: An NCHW tensor of type float16, float32, int8.
    pool_op.set_input_x(relu1);
    // mode: either "1" (max pooling) or "0" (avg pooling). Defaults to "0".
    pool_op.set_attr_mode(0);
    // window[0]: int32, specifying the window size along in the H dimension. The value range is [1, 32768]. Defaults to "1".
    // window[1]: int32, specifying the window size along in the W dimension. The value range is [1, 32768]. Defaults to "1".
    pool_op.set_attr_window({4, 4});
    // stride[0]: An optional int32, specifying the stride along in the H dimension. The value range is [1, 63]. Defaults to "1".
    // stride[1]: An optional int32, specifying the stride along in the W dimension. The value range is [1, 63]. Defaults to "1".
    pool_op.set_attr_stride({4, 4});
    // pads: A list of 4 integers. Specifying the top, bottom, left and right
    pool_op.set_attr_pad({ 0, 0, 0, 0});
    // dilation: A list of 4 integers. Specifying the up, bottom, left and right
    pool_op.set_attr_dilation({1, 1, 1, 1});
    // ceil_mode: int32, either "0" (ceil mode) or "1" (floor mode). Defaults to "0".
    pool_op.set_attr_ceil_mode(0);
    // update tensor input
    ge::TensorDesc pooling_input_desc_x(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorDesc pooling_output_desc_y(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
    pool_op.update_input_desc_x(pooling_input_desc_x);
    pool_op.update_output_desc_y(pooling_output_desc_y);

    // Build Graph
    std::vector<ge::Operator> inputs{ input_x };
    std::vector<ge::Operator> outputs{ pool_op };
    std::vector<std::pair<ge::Operator, std::string>> outputs_with_name = {{ pool_op, "y" }};

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
    options.insert(std::make_pair(ge::ir_option::LOG_LEVEL, "debug"));
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