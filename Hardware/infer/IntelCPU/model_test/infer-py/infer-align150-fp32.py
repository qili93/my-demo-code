from __future__ import print_function

import os
import numpy as np
import cv2
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid import core

image_path="../assets/images/face.jpg"
imgnp_path="../assets/images/face.raw"
detect_path="../assets/images/face_detect.jpg"
model_path="../assets/models/align150-fp32"

paddle.enable_static()

# # change all depthwise_conv2d => conv2d
# def _prepare_for_fp32_mkldnn(graph):
#     ops = graph.all_op_nodes()
#     for op_node in ops:
#         name = op_node.name()
#         if name in ['depthwise_conv2d']:
#             input_var_node = graph._find_node_by_name(op_node.inputs, op_node.input("Input")[0])
#             weight_var_node = graph._find_node_by_name(op_node.inputs, op_node.input("Filter")[0])
#             output_var_node = graph._find_node_by_name(graph.all_var_nodes(), op_node.output("Output")[0])
#             attrs = {
#                 name: op_node.op().attr(name)
#                 for name in op_node.op().attr_names()
#             }
#             conv_op_node = graph.create_op_node(
#                 op_type='conv2d',
#                 attrs=attrs,
#                 inputs={
#                     'Input': input_var_node,
#                     'Filter': weight_var_node
#                 },
#                 outputs={'Output': output_var_node})
#             graph.link_to(input_var_node, conv_op_node)
#             graph.link_to(weight_var_node, conv_op_node)
#             graph.link_to(conv_op_node, output_var_node)
#             graph.safe_remove_nodes(op_node)
#     return graph

def drawRotateRectange(img, face, color, thickness):
    xmin = int(face[2] * img.shape[1])
    # print(xmin)
    ymin = int(face[3] * img.shape[0])
    # print(ymin)
    xmax = int(face[4] * img.shape[1])
    # print(xmax)
    ymax = int(face[5] * img.shape[0])
    # print(ymax)
    cv2.line(img,(xmin, ymin), (xmin, ymax), color, thickness)
    cv2.line(img,(xmin, ymin), (xmax, ymin), color, thickness)
    cv2.line(img,(xmax, ymax), (xmin, ymax), color, thickness)
    cv2.line(img,(xmax, ymax), (xmax, ymin), color, thickness)
    score = '%.3f'%(face[1])
    cv2.putText(img, str(score), (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0 ,0), thickness = 1, lineType = 8)


def draw_image(image_path, detection):
    image = cv2.imread(image_path)
    dect_conf_threshold = 0.2
    for i in range(len(detection)):
        if detection[i][1] >= dect_conf_threshold:
            drawRotateRectange(image, detection[i],(0,255,0),2)
    cv2.imwrite(detect_path, image)

def read_image(image_path):
    image = cv2.imread(image_path)
    print("original image.shape=",image.shape) # height, width, channel
    scale = 320.0 / image.shape[0]
    new_height = int(image.shape[0] * scale)
    new_width = int(image.shape[1] * scale)
    image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    print("resized image.shape=",image.shape)
    img_np = np.array(image)
    if len(img_np.shape) == 3:
        img_np = np.swapaxes(img_np, 1, 2)
        img_np = np.swapaxes(img_np, 1, 0)
    mean = [104., 117., 123.]
    scale = 0.007843
    img_np = img_np.astype('float32')
    img_np -= np.array(mean)[:, np.newaxis, np.newaxis].astype('float32')
    img_np = img_np * scale
    img_np = [img_np] # expand axis = 0
    img_np = np.array(img_np)
    print("img_np.shape=", img_np.shape)
    with open(imgnp_path, "wb") as f:
        img_np.tofile(f)
    return img_np

def read_rawfile(imgnp_path):
    img_np = np.fromfile(imgnp_path, dtype=np.float32)
    img_np.resize(1, 3, 320, 512)
    print("img_np.shape=", img_np.shape)
    return img_np

def infer_model(model_path):
    if model_path is None:
        return

    img_np = np.ones([1, 3, 128, 128]).astype('float32')
    # img_np = read_image(image_path)
    # img_np = read_rawfile(imgnp_path)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.executor.global_scope()
    with fluid.scope_guard(inference_scope):
      [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
                                                 model_path, exe, '__model__', '__params__')

    #   graph = IrGraph(core.Graph(inference_program.desc), for_test=True)
    #   graph = _prepare_for_fp32_mkldnn(graph)
    #   inference_program = graph.to_program()

      detection, = exe.run(inference_program,
                    feed={feed_target_names[0]: img_np},
                    fetch_list=fetch_targets,
                    return_numpy=False)
                    
      detection = np.array(detection)
      print("detection=", detection)

      draw_image(image_path, detection)

if __name__ == '__main__':
    infer_model(model_path='../assets/models/align150-fp32')