from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid

def x2paddle_net():
    x2paddle_input = fluid.layers.data(dtype='float32', shape=[-1, 3, 2, 2], name='x2paddle_input', append_batch_size=False)
    x2paddle_output = fluid.layers.conv2d(x2paddle_input, num_filters=3, filter_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=3, param_attr='x2paddle_conv1_weight', name='x2paddle_output', bias_attr='x2paddle_conv1_bias')

    return [x2paddle_input], [x2paddle_output]

def run_net(param_dir="./"):
    import os
    inputs, outputs = x2paddle_net()

    ops = fluid.default_main_program().global_block().ops
    used_vars = list()
    for op in ops:
        used_vars += op.input_arg_names

    tmp = list()
    for input in inputs:
        if isinstance(input, list):
            for ipt in input:
                if ipt.name not in used_vars:
                    continue
                tmp.append(ipt)
        else:
            if input.name not in used_vars:
                continue
            tmp.append(input)
    inputs = tmp
    for i, out in enumerate(outputs):
        if isinstance(out, list):
            for out_part in out:
                outputs.append(out_part)
            del outputs[i]
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    def if_exist(var):
        b = os.path.exists(os.path.join(param_dir, var.name))
        return b

    fluid.io.load_vars(
        exe, param_dir, fluid.default_main_program(), predicate=if_exist)
