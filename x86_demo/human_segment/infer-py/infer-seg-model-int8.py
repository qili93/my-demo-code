from __future__ import print_function

import os
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid import core

paddle.enable_static()

# change all depthwise_conv2d => conv2d
def _prepare_for_fp32_mkldnn(graph):
    ops = graph.all_op_nodes()
    for op_node in ops:
        name = op_node.name()
        if name in ['depthwise_conv2d']:
            input_var_node = graph._find_node_by_name(op_node.inputs, op_node.input("Input")[0])
            weight_var_node = graph._find_node_by_name(op_node.inputs, op_node.input("Filter")[0])
            output_var_node = graph._find_node_by_name(graph.all_var_nodes(), op_node.output("Output")[0])
            attrs = {
                name: op_node.op().attr(name)
                for name in op_node.op().attr_names()
            }
            conv_op_node = graph.create_op_node(
                op_type='conv2d',
                attrs=attrs,
                inputs={
                    'Input': input_var_node,
                    'Filter': weight_var_node
                },
                outputs={'Output': output_var_node})
            graph.link_to(input_var_node, conv_op_node)
            graph.link_to(weight_var_node, conv_op_node)
            graph.link_to(conv_op_node, output_var_node)
            graph.safe_remove_nodes(op_node)
    return graph

def infer_model(model_path):
    if model_path is None:
        return

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.executor.global_scope()
    with fluid.scope_guard(inference_scope):
      [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe, '__model__', '__params__')

      graph = IrGraph(core.Graph(inference_program.desc), for_test=True)
      graph = _prepare_for_fp32_mkldnn(graph)

      inference_program = graph.to_program()

      # change the input shape based on models
      images = np.ones([1, 4, 192, 192]).astype('float32')
      out = exe.run(inference_program,
                    feed={feed_target_names[0]: images},
                    fetch_list=fetch_targets)
      print(out[0])

if __name__ == '__main__':
    infer_model(model_path='../assets/models/seg-model-int8')