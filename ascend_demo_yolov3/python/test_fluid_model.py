# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Author: Drift
Email:  wutuobang@baidu.com
Date:   2019/5/28 下午4:19
"""

import argparse
import os
import datetime

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import cv2
import numpy as np


# 指定运行哪个model
cur_dir = os.path.dirname(os.path.abspath(__file__))
image_file = cur_dir + '/../assets/models/image/stop.jpg'
model_define = cur_dir + '/../assets/models/yolov3_model/model'
model_weight = cur_dir + '/../assets/models/yolov3_model/params'
model_labels = cur_dir + '/../assets/models/yolov3_model/label_list.txt'
conf_threshold = 0.3
infer_size = 608


class MyPredictor:
    """ predictor """
    def __init__(self,
                 model_file='model',
                 param_file='params',
                 label_file='label_list.txt'):
        """

        :param model_file:
        :param param_file:
        :param label_file:
        """
        self._model_dir = os.path.dirname(os.path.abspath(model_file))
        self._model_file = model_file
        self._param_file = param_file
        self._label_file = label_file
        self._get_numpy_ret = False

    def infer(self, test_image, conf_thresh, top_num=200):
        """

        :param test_image:
        :param conf_thresh:
        :param top_num:
        :return:
        """
        from paddle.fluid.core import PaddleBuf
        from paddle.fluid.core import PaddleDType
        from paddle.fluid.core import PaddleTensor
        from paddle.fluid.core import AnalysisConfig
        from paddle.fluid.core import create_paddle_predictor

        # 预处理
        data = self.preprocess_img(test_image)
        data = np.expand_dims(data, 0)

        shape = [np.array([608, 608], dtype=np.int32)]
        shape = np.expand_dims(shape, 0)

        # 设置AnalysisConfig
        config = AnalysisConfig(self._model_file, self._param_file)
        # config.enable_use_gpu(2000, 0)

        # 创建PaddlePredictor
        predictor = create_paddle_predictor(config)

        # 设置输入
        image = PaddleTensor(data.copy().astype(np.float32))
        # image = PaddleTensor()
        image.name = "data"
        image.shape = [1, 3, 608, 608]
        image.dtype = PaddleDType.FLOAT32
        # image.data = PaddleBuf(data.flatten().astype(np.float32).tolist())

        im_size = PaddleTensor(shape.copy().astype(np.int32))
        im_size.name = "im_size"
        im_size.shape = [1, 2]
        im_size.dtype = PaddleDType.INT32
        # im_size.data = PaddleBuf(shape.flatten().astype(np.int32).tolist())
        inputs = [image, im_size]

        # 运行预测引擎
        t1 = datetime.datetime.now()
        results = predictor.run(inputs)
        t2 = datetime.datetime.now()
        print('infer time is %f milliseconds' % ((t2 - t1).total_seconds() * 1000))

        # 后处理
        output_data = results[0].data.float_data()
        output_shape = results[0].shape
        self.post_process(output_data, output_shape, top_num, test_image, conf_thresh)

    def preprocess_img(self, test_image):
        """

        :param test_image:
        :return:
        """
        img = cv2.imread(test_image)

        # BGR2RGB
        img = img[..., ::-1]

        # resize
        # h, w, _ = img.shape
        # im_scale_x = infer_size / float(w)
        # im_scale_y = infer_size / float(h)
        # img = cv2.resize(
        #     img,
        #     None,
        #     None,
        #     fx=im_scale_x,
        #     fy=im_scale_y,
        #     interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)

        # channel_first
        img = np.transpose(img, (2, 0, 1))

        # mean
        mean = [123.675, 116.28, 103.53]
        img_mean = np.array(mean)[:, np.newaxis, np.newaxis].astype(np.float32)
        img = img - img_mean

        # scale
        scale = [0.0171247538317, 0.0175070028011, 0.0174291938998]
        img_scale = np.array(scale)[:, np.newaxis, np.newaxis].astype(np.float32)
        img = img * img_scale

        return img

    def post_process(self, results, shape, top_num, test_image, conf_thresh):
        """

        :param results:
        :param shape:
        :param top_num:
        :param test_image:
        :param conf_thresh:
        :return:
        """
        ret = np.array(results).reshape(shape)
        idx = ret[:, 1].argsort()[-top_num:][::-1]
        mat = ret[idx]

        labels = {}
        with open(self._label_file) as inf:
            for idx, line in enumerate(inf.readlines()):
                line = line.strip()
                labels[idx] = line

        res_img = Image.open(test_image)
        draw = ImageDraw.Draw(res_img)
        ori_width, ori_height = res_img.size

        print('detect result is:')
        result = []
        for item in mat:
            label_idx = int(item[0])
            score = float(item[1])
            x1 = float(item[2])
            y1 = float(item[3])
            x2 = float(item[4])
            y2 = float(item[5])

            if score < conf_thresh:
                continue

            result.append({
                'score': score,
                'name': labels[label_idx],
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
            })
            print(label_idx, '\t', score, '\t',
                  item[2], '\t', item[3], '\t', item[4], '\t', item[5], '\t', labels[label_idx])

        font = ImageFont.truetype('/usr/share/fonts/msttcore/timesbd.ttf', size=20, encoding='unic')

        for res in result:
            x_min = int(round(res['x1']) * ori_width / infer_size)
            y_min = int(round(res['y1']) * ori_height / infer_size)
            x_max = int(round(res['x2']) * ori_width / infer_size)
            y_max = int(round(res['y2']) * ori_height / infer_size)

            draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=2)
            text = str(res['name']) + ':' + str(res['score'])
            draw.text([x_min, y_max], text, (0, 0, 255), font)

        res_img.save(test_image + '.fluid.result.jpg')
        print("save new image file to {}".format(test_image + '.fluid.result.jpg'))
        
        return True


def infer(args):
    """

    :param args:
    :return:
    """
    predictor_ins = MyPredictor(args.model_def, args.model_weights, args.label_map)

    print("test image file is {}".format(args.image_file))
    predictor_ins.infer(args.image_file, args.conf_thresh)

def parse_args():
    """
    args
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', default=image_file)
    parser.add_argument('--model_def', default=model_define)
    parser.add_argument('--model_weights', default=model_weight)
    parser.add_argument('--label_map', default=model_labels)
    parser.add_argument('--conf_thresh', default=conf_threshold)

    return parser.parse_args()


if __name__ == '__main__':
    infer(parse_args())
