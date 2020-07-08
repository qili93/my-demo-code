#include "graph/graph.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "graph/ge_error_codes.h"
#include "ge/ge_api_types.h"
#include "ge/ge_ir_build.h"
#include "all_ops.h" // opp/op_proto/built-in/inc

#include "model_build.h"

bool OMModelBuild::GenGraph(ge::Graph& graph) {
     // // input data op => feed
    ge::TensorDesc input_desc(ge::Shape({ 1, 1, 4, 4 }), ge::FORMAT_ND, ge::DT_FLOAT);
    auto input_x = ge::op::Data("input_x");
    input_x.update_input_desc_x(input_desc);
    input_x.update_output_desc_y(input_desc);

    auto conv_op = ge::op::Conv2D("conv1")
         .set_input_x();

    auto relu1 = ge::op::Relu("relu")
        .set_input_x(input_x, "y");

    // Build Graph
    std::vector<ge::Operator> inputs{ input_x };
    std::vector<ge::Operator> outputs{ relu1 };
    std::vector<std::pair<ge::Operator, std::string>> outputs_with_name = {{relu1, "y"}};

    graph.SetInputs(inputs).SetOutputs(outputs);
    return true;
}

bool OMModelBuild::SaveModel(ge::Graph& om_graph, std::string model_path)
{
    INFO_LOG("-------Enter: [model_build](SaveModel)-------");
    // 1. Genetate graph
    // ge::Graph om_graph("bias_add_graph");
    // if(!GenGraph(om_graph)) {
    //   ERROR_LOG("Generate BiasAdd Graph Failed!");
    // }
    // INFO_LOG("Generate BiasAdd Graph SUCCESS!");

    // 2. system init
    std::map<std::string, std::string> global_options = {
        {ge::ir_option::SOC_VERSION, "Ascend310"},
    };
    if (ge::aclgrphBuildInitialize(global_options) !=  ge::GRAPH_SUCCESS) {
      ERROR_LOG("[model_build](SaveModel) aclgrphBuildInitialize Failed!");
    } else {
        INFO_LOG("[model_build](SaveModel) aclgrphBuildInitialize succees");
    }

    // 3. Build IR Model
    ge::ModelBufferData model_om_buffer;
    std::map<std::string, std::string> options;
    //PrepareOptions(options);

    if (ge::aclgrphBuildModel(om_graph, options, model_om_buffer) !=  ge::GRAPH_SUCCESS) {
      ERROR_LOG("[model_build](SaveModel) aclgrphBuildModel Failed!");
    } else {
        INFO_LOG("[model_build](SaveModel) aclgrphBuildModel succees");
    }

    // 4. Save IR Model
    if (ge::aclgrphSaveModel(model_path, model_om_buffer) != ge::GRAPH_SUCCESS) {
      ERROR_LOG("[model_build](SaveModel) aclgrphSaveModel Failed!");
    } else {
        INFO_LOG("[model_build](SaveModel) aclgrphSaveModel succees");
    }

    // 5. release resource
    ge::aclgrphBuildFinalize();
    INFO_LOG("-------Leave: [model_build](SaveModel)-------");
    return true;
}