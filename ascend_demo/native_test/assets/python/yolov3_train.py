#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import os
import shutil
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import TracedLayer

class TestYoloV3Net(fluid.dygraph.Layer):
  def __init__(self):
    super(TestYoloV3Net, self).__init__()
    self.initTestCase()

    self.conv1 = fluid.dygraph.Conv2D(num_channels=1, num_filters=21, filter_size=10, stride=1)

  def initTestCase(self):
    self.anchors_low = [116, 90, 156, 198, 373, 326] # low
    self.anchors_mid = [30, 61, 62, 45, 59, 119] # mid
    self.anchors_high = [10, 13, 16, 30, 33, 23] # high
    an_num = int(len(self.anchors_low) // 2) # 3
    # self.batch_size = 32 # low
    self.class_num = 2
    self.conf_thresh = 0.5
    self.downsample_low = 32
    self.downsample_mid = 16 # mid
    self.downsample_high = 8 # high
    # self.x_shape_low = (self.batch_size, 21, 19, 19)
    # self.x_shape_mid = (self.batch_size, 21, 38, 38)
    # self.x_shape_high = (self.batch_size, 21, 76, 76)
    # self.imgsize_shape = (self.batch_size, 2) # n, 2
    # dim_x[1] == anchor_num * (5 + class_num)

  def forward(self, input_low, input_mid, input_high, img_size):
    # # [-1, 1, 28, 28]
    # input_low = self.conv1(input_data)
    # # [-1, 21, 19, 19]
    # input_mid = fluid.layers.resize_nearest(input_low, scale=2)
    # # [-1, 21, 38, 38]
    # input_high = fluid.layers.resize_nearest(input_mid, scale=2)
    # [-1, 21, 76, 76]
    boxes_low, scores_low = fluid.layers.yolo_box(x=input_low, img_size=img_size, 
                                                  class_num=self.class_num, 
                                                  anchors=self.anchors_low, 
                                                  conf_thresh=self.conf_thresh, 
                                                  downsample_ratio=self.downsample_low)
    boxes_mid, scores_mid = fluid.layers.yolo_box(x=input_mid, img_size=img_size, 
                                                  class_num=self.class_num, 
                                                  anchors=self.anchors_mid, 
                                                  conf_thresh=self.conf_thresh, 
                                                  downsample_ratio=self.downsample_mid)
    boxes_high, scores_high = fluid.layers.yolo_box(x=input_high, img_size=img_size,
                                                  class_num=self.class_num, 
                                                  anchors=self.anchors_high, 
                                                  conf_thresh=self.conf_thresh, 
                                                  downsample_ratio=self.downsample_high)

    concat_0_out = fluid.layers.concat(input=[boxes_low, boxes_mid, boxes_high], axis=1)

    transpose2_low = paddle.fluid.layers.transpose(scores_low, perm=[0,2,1])
    transpose2_mid = paddle.fluid.layers.transpose(scores_mid, perm=[0,2,1])
    transpose2_high = paddle.fluid.layers.transpose(scores_high, perm=[0,2,1])

    concat_1_out = fluid.layers.concat(input=[transpose2_low, transpose2_mid, transpose2_high], axis=2)

    out = fluid.layers.multiclass_nms(bboxes=concat_0_out, scores=concat_1_out, 
                                      score_threshold=0.01, nms_top_k=100, keep_top_k=16, nms_threshold=0.45, 
                                      normalized=False, nms_eta=1.0, background_label=0)
    return out

def train_yolov3(num_epochs, save_dirname):
    place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()

    with fluid.dygraph.guard(place):
        yolo_v3_net = TestYoloV3Net()

        adam = fluid.optimizer.AdamOptimizer(
            learning_rate=0.001, parameter_list=yolo_v3_net.parameters())

        input_data_low = np.random.random((BATCH_SIZE, 21, 6, 6)).astype('float32')
        input_data_mid = np.random.random((BATCH_SIZE, 21, 12, 12)).astype('float32')
        input_data_high = np.random.random((BATCH_SIZE, 21, 24, 24)).astype('float32')
        input_data_imgsize = np.random.randint(10, 20, (BATCH_SIZE, 2)).astype('int32')

        input_var_low = fluid.dygraph.base.to_variable(input_data_low)
        input_var_mid = fluid.dygraph.base.to_variable(input_data_mid)
        input_var_high = fluid.dygraph.base.to_variable(input_data_high)
        input_var_imgsize = fluid.dygraph.base.to_variable(input_data_imgsize)

        for epoch in range(num_epochs):
            output = yolo_v3_net(input_var_low, input_var_mid, input_var_high, input_var_imgsize)
            # print(output.numpy().shape)

            label_data = np.random.random(output.numpy().shape[0]).astype('int64')
            label_var = fluid.dygraph.to_variable(label_data)
            loss = fluid.layers.cross_entropy(output, label_var)

            avg_loss = fluid.layers.mean(loss)
            avg_loss.backward()
            adam.minimize(avg_loss)

            yolo_v3_net.clear_gradients()

        # save inference model
        if save_dirname is None:
            return
        # delete old model
        if  os.path.exists(save_dirname):
            shutil.rmtree(save_dirname)
            os.makedirs(save_dirname)
        # save inference model
        out_dygraph, static_layer = TracedLayer.trace(yolo_v3_net, inputs=[input_var_low, input_var_mid, input_var_high, input_var_imgsize])
        
        static_layer.save_inference_model(save_dirname, feed=range(4), fetch=[0])
        print("Saved inference model to {}".format(save_dirname))

if __name__ == '__main__':
    BATCH_SIZE = 1
    train_yolov3(num_epochs=1, save_dirname='./yolov3_model')