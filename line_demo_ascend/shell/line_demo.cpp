#include <iostream>
#include <fstream>
#include <algorithm>
#include <string.h>
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_ir_build.h"
#include "all_ops.h"
#include <dlfcn.h>
#include <unistd.h>

using namespace std;
using namespace ge;
using ge::Operator;

void PrepareOptions(std::map<std::string, std::string>& options) {
}

bool GenGraph(Graph& graph) {
    // input data op => feed
    TensorDesc feed_desc(ge::Shape({ 2, 2 }), FORMAT_ND, DT_INT8);
    auto feed = op::Data("feed");
    feed.update_input_desc_x(feed_desc);
    feed.update_output_desc_y(feed_desc);

    // const data op => bias
    TensorDesc bias_desc(ge::Shape({ 2 }), FORMAT_ND, DT_INT8);
    uint8_t *bias_data = new uint8_t[2];
    for (int i = 0; i < 2; ++i) {
        *(bias_data+i) = 1;
    }
    Tensor bias_tensor(bias_desc, (uint8_t*)bias_data, 2*sizeof(uint8_t))
    auto bias1 = op::Const("Add/bias").set_attr_value(bias_tensor);

    // bias add op
    auto bias_add = op::BiasAdd("bias_add")
        .set_input_x1(feed)
        .set_input_x2(bias1)
        .set_attr_data_format("ND");

    // Build Graph
    std::vector<Operator> inputs{ feed };
    std::vector<Operator> outputs{ bias_add };

    graph.SetInputs(inputs).SetOutputs(outputs);
    return true;
}

bool SaveOMModel() {
    // 1. generate graph
    Graph om_graph("bias_add_graph");
    if (!GenGraph(om_graph)) {
        cout << "Generate BiasAdd Graph Failed!" << endl;
        return false;
    }
    // 2. system init
    std::map<std::string, std::string> global_options = {{ge::ir_option::SOC_VERSION, "Ascend310"},};
    auto status = aclgrphBuildInitialize(global_options);

    // 3. Build IR Model
    ModelBufferData model_om_buffer;
    std::map<std::string, std::string> options;
    PrepareOptions(options);
    status = aclgrphBuildModel(om_graph, options, model_om_buffer);
    if (status == GRAPH_SUCCESS) {
        cout << "Build BiasAdd Model SUCCESS!" << endl;
    }
    else {
        cout << "Build BiasAdd Model Failed!" << endl;
    }

    // 4. Save IR Model
    status = aclgrphSaveModel("bias_add_model", model_om_buffer);
    if (status == GRAPH_SUCCESS) {
        cout << "Save Offline BiasAdd Model SUCCESS!" << endl;
    }
    else {
        cout << "Save Offline BiasAdd Model Failed!" << endl;
    }

    // 5. release resource
    aclgrphBuildFinalize();

    return true;
}

int main(int argc, char* argv[])
{
    if (SaveOMModel()) {
        cout << "Save BiasAdd OM Model SUCCESS!" << endl;
    }
    return 0;
}
