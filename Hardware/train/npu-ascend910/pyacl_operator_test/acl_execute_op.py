# -*- coding:utf-8 -*-
import numpy as np
import os
import acl

from constant import ACL_MEMCPY_HOST_TO_DEVICE, \
    ACL_MEM_MALLOC_NORMAL_ONLY, ACL_FORMAT_NCHW, ACL_FORMAT_ND, \
    acl_dtype, NPY_INT, ACL_ERROR_NONE


def check_ret(message, ret):
    if ret != ACL_ERROR_NONE:
        raise Exception("{} failed ret={}"
                        .format(message, ret))


class Operator():
    def __init__(self,
                 operator_input,
                 operator_output,
                 operator_attr,
                 device_id=0,
                 op_model_path="./op_models",
                 format_type=ACL_FORMAT_ND,
                 config_path=None,
                 op_type="ResizeNearestNeighborV2"):
        self.device_id = device_id  # int
        self.op_model_path = op_model_path  # string
        self.config_path = config_path
        self.context = None  # pointer
        self.stream = None
        self.op_type = op_type
        self.format_type = format_type
        self.operator_input = operator_input
        self.operator_output = operator_output
        self.operator_attr = operator_attr
        self.op_attr = None

        self._inputs_desc = []
        self._inputs_device = []
        self._inputs_device_buffer = []
        self.host_inputs = []

        self.output_desc = []
        self.device_outputs = []
        self.device_buffer_outputs = []
        self.host_outputs = []
        self.init_resource()

    def __del__(self):
        print('release source stage:')
        while self._inputs_desc:
            ret = acl.destroy_data_buffer(self._inputs_device_buffer.pop())
            check_ret("acl.destroy_data_buffer", ret)
            ret = acl.rt.free(self._inputs_device.pop())
            check_ret("acl.rt.free", ret)
            acl.destroy_tensor_desc(self._inputs_desc.pop())

        while self.output_desc:
            ret = acl.destroy_data_buffer(self.device_buffer_outputs.pop())
            check_ret("acl.destroy_data_buffer", ret)
            ret = acl.rt.free(self.device_outputs.pop())
            check_ret("acl.rt.free", ret)
            acl.destroy_tensor_desc(self.output_desc.pop())

        if self.op_attr:
            acl.op.destroy_attr(self.op_attr)
            self.op_attr = None

        if self.stream:
            ret = acl.rt.destroy_stream(self.stream)
            check_ret("acl.rt.destroy_stream", ret)
            self.stream = None

        if self.context:
            ret = acl.rt.destroy_context(self.context)
            check_ret("acl.rt.destroy_context", ret)
            self.context = None

        ret = acl.rt.reset_device(self.device_id)
        check_ret("acl.rt.reset_device", ret)
        ret = acl.finalize()
        check_ret("acl.finalize", ret)
        print('release source success')

    def init_resource(self):
        print("init resource stage:")
        if isinstance(self.config_path, str) \
                and os.path.exists(self.config_path):
            ret = acl.init(self.config_path)
            check_ret("acl.init", ret)
        elif self.config_path is None:
            ret = acl.init()
            check_ret("acl.init", ret)
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        check_ret("acl.rt.create_stream", ret)

        # self.shape = self.factor_a.shape
        # if self.factor_b.shape != self.shape:
        #     raise ValueError("factor_a:{} factor_b:{} isn't same shape!!!"
        #                      .format(self.shape, self.factor_b.shape))
        # self.shape = list(self.shape)
        # self.data_type = str(self.factor_a.dtype)
        # if str(self.factor_b.dtype) != self.data_type:
        #     raise ValueError("factor_a:{} factor_b:{} isn't same dtype!!!"
        #                      .format(self.factor_a.dtype, self.factor_b.dtype))

        ret = acl.op.set_model_dir(self.op_model_path)
        check_ret("acl.op.set_model_dir", ret)
        print("init resource success")

    def _gen_input_tensor(self):
        print("gen input data stage:")
        for input_item in self.operator_input:
            input_shape = list(input_item.shape)
            input_dtype = str(input_item.dtype)
            input_format = ACL_FORMAT_NCHW if len(input_item.shape) == 4 else ACL_FORMAT_ND
            tensor = acl.create_tensor_desc(acl_dtype[input_dtype], input_shape, input_format)
            factor_size = acl.get_tensor_desc_size(tensor)
            factor_device, ret = acl.rt.malloc(
                factor_size, ACL_MEM_MALLOC_NORMAL_ONLY)
            check_ret("acl.rt.malloc", ret)
            factor_ptr = acl.util.numpy_to_ptr(input_item)

            ret = acl.rt.memcpy(factor_device,
                                factor_size,
                                factor_ptr,
                                factor_size,
                                ACL_MEMCPY_HOST_TO_DEVICE)
            check_ret("acl.rt.memcpy", ret)
            factor_buffer = acl.create_data_buffer(factor_device, factor_size)
            self._inputs_device.append(factor_device)
            self._inputs_device_buffer.append(factor_buffer)
            self._inputs_desc.append(tensor)
        print("gen input data success")

    def _gen_output_tensor(self):
        print("gen output data stage:")
        for output_item in self.operator_output:
            output_shape = list(output_item.shape)
            ouptut_dtype = str(output_item.dtype)
            output_format = ACL_FORMAT_NCHW if len(output_item.shape) == 4 else ACL_FORMAT_ND
            out_tensor = acl.create_tensor_desc(acl_dtype[ouptut_dtype], output_shape, output_format)
            factor_size = acl.get_tensor_desc_size(out_tensor)
            factor_device, ret = acl.rt.malloc(
                factor_size, ACL_MEM_MALLOC_NORMAL_ONLY)
            check_ret("acl.rt.malloc", ret)
            self.device_outputs.append(factor_device)
            self.device_buffer_outputs.append(
                acl.create_data_buffer(factor_device, factor_size)
            )
            self.host_outputs.append(acl.rt.malloc_host(factor_size)[0])
            self.output_desc.append(out_tensor)
            print("gen output data success")

    def _gen_operator_attr(self):
        print("gen operator attr stage:")
        self.op_attr = acl.op.create_attr()
        for key, value in self.operator_attr.items():
            ret = acl.op.set_attr_bool(self.op_attr, key, value)
            check_ret("acl.op.set_attr_int", ret)


    def _infer_shape(self):
        print("infer shape stage: ", len(self.output_desc))
        ret = acl.op.infer_shape(
            self.op_type,
            self._inputs_desc, 
            self._inputs_device_buffer,
            len(self.output_desc),
            self.output_desc,
            self.op_attr)
        tensor_dims = []
        for i in range(len(self.output_desc)):
            dim_nums = acl.get_tensor_desc_num_dims(self.output_desc[i])
            dim_size = []
            for j in range(dim_nums):
                dim, ret = acl.get_tensor_desc_dim_v2(self.output_desc[i], j)
                print("0: dim = ", dim)
                if dim == -1:
                    dim_range, ret = acl.get_tensor_desc_dim_range(self.output_desc[i], j, 2)
                    dim = dim_range[1]
                print("1: dim = ", dim)
                dim_size.append(dim)
            tensor_dims.append(dim_size)
        print(tensor_dims)


    def run(self):
        self._gen_input_tensor()
        self._gen_output_tensor()
        self._gen_operator_attr()
        self._infer_shape()
        self._forward()
        result = self._get_operator_result()
        return result


    def _forward(self):
        print('execute stage:')
        ret = acl.op.execute_v2(
            self.op_type,
            self._inputs_desc,
            self._inputs_device_buffer,
            self.output_desc,
            self.device_buffer_outputs,
            self.op_attr,
            self.stream)
        check_ret("acl.op.execute_v2", ret)
        ret = acl.rt.synchronize_stream(self.stream)
        check_ret("acl.rt.synchronize_stream", ret)
        print('execute success')

    def _get_operator_result(self):
        print('get operator result stage:')
        result = []
        for index in range(len(self.output_desc)):
            factor = self.output_desc[index]
            factor_size = acl.get_tensor_desc_size(factor)
            ret = acl.rt.memcpy(self.host_outputs[index],
                                factor_size,
                                self.device_outputs[index],
                                factor_size,
                                ACL_MEMCPY_HOST_TO_DEVICE)
            check_ret("acl.rt.memcpy", ret)

            print("shape:", tuple(self.shape))
            data = acl.util.ptr_to_numpy(self.host_outputs[index],
                                         tuple(self.shape),
                                         NPY_INT)
            print("ACL output:\n", data)
            result.append(data)
        print('get operator result success')
        return result


if __name__ == '__main__':
    input_x = np.array([[1, 2, 3],[4, 5, 6]]).reshape((1, 1, 2, 3)).astype(np.float32)
    input_size = np.array([3, 3]).astype(np.int32)
    output_y = np.zeros((1, 1, 3, 3)).astype(np.float32)

    input_list = [input_x, input_size]
    output_list = [output_y]
    attr_dict = {"align_corners" : False, "half_pixel_centers" : False}

    print("input_x.shape = ", input_x.shape)
    print("input_x.data = \n", input_x)
    print("input_size.shape = ", input_size.shape)
    print("input_size.data = \n", input_size)
    print("output_y.shape = ", output_y.shape)
    print("output_y.data = \n", output_y)

    op = Operator(input_list, output_list, attr_dict)
    result_list = op.run()
