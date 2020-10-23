# coding=utf-8
import os
import time
from PIL import Image
import numpy as np
import paddle
import paddle.fluid as fluid
import numpy as np
import sys
import cv2
import time

paddle.enable_static()


class Detection:
    def __init__(self, model_path='./detection/faceboxes_origin', model_filename='model', params_filename='weights', use_gpu=False):
        self.place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)
        self.detection_scope = fluid.Scope()
        with fluid.scope_guard(self.detection_scope):
            [self.inference_program, self.feed_target_names, self.fetch_targets] = \
                fluid.io.load_inference_model(
                model_path,
                self.exe,
                model_filename=model_filename,
                params_filename=params_filename)

    def load_image(self, file):
        im = Image.open(file)
        im = im.resize((512, 512), Image.ANTIALIAS)
        im = np.array(im).astype(np.float32)

        im = im.transpose((2, 0, 1))
        im = im[(2, 1, 0), :, :]  # BGR

        mean = [123.68, 116.78, 103.94]
        mean = np.array(mean, dtype=np.float32)
        mean = mean[:, np.newaxis, np.newaxis]
        print(mean.shape, im.shape)
        im -= mean

        return im

    def to_chw_bgr(self, image):
        """
        Transpose image from HWC to CHW and from RBG to BGR.
        Args:
            image (np.array): an image with HWC and RBG layout.
        """
        # HWC to CHW
        if len(image.shape) == 3:
            image = np.swapaxes(image, 1, 2)
            image = np.swapaxes(image, 1, 0)
        # RBG to BGR
        #image = image[[2, 1, 0], :, :]
        return image

    def draw_image(self, faces_pred, img, dect_conf_threshold):
        for i in range(len(faces_pred)):
            #print(faces_pred)
            if faces_pred[i][4] >= dect_conf_threshold:
                self.drawRotateRectange(img,faces_pred[i],(0,255,0),1)

    def drawRotateRectange(self, img, face, color, thickness):
        cv2.line(img,(int(face[0]),int(face[1])), (int(face[2]),int(face[1])),color,thickness)
        cv2.line(img,(int(face[2]),int(face[1])), (int(face[2]),int(face[3])),color,thickness)
        cv2.line(img,(int(face[0]),int(face[1])), (int(face[0]),int(face[3])),color,thickness)
        cv2.line(img,(int(face[2]),int(face[3])), (int(face[0]),int(face[3])),color,thickness)
        score = '%.3f'%(face[4])
        cv2.putText(img, str(score), (int((face[0]+face[2])/2),int((face[1]+face[3])/2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0 ,0), thickness = 1, lineType = 8)

    def infer_one(self, image):
        shrink = 1.0
        image_shape = [3, image.shape[0], image.shape[1]]
        if shrink != 1:
            h, w = int(image_shape[1] * shrink), int(image_shape[2] * shrink)
            image = image.resize((w, h), Image.ANTIALIAS)
            image_shape = [3, h, w]

        # change here 
        # img = cv2.resize(image, (320, 240))
        
        img = np.array(image)
        img = self.to_chw_bgr(img)
        mean = [104., 117., 123.]
        scale = 0.007843

        img = img.astype('float32')
        img -= np.array(mean)[:, np.newaxis, np.newaxis].astype('float32')

        img = img * scale
        img = [img]
        img = np.array(img)

        total_time = 0
        total_num = 100
        #print('DEBUG')

        print("img.shape=", img.shape)

        '''
        # img = np.ones([1, 3, 512, 512]).astype('float32') 
        with fluid.scope_guard(self.detection_scope):
            for i in range(1):
                t1 = time.time()
                detection, = self.exe.run(self.inference_program,
                                  feed={self.feed_target_names[0]:img},
                                  fetch_list=self.fetch_targets,
                                  return_numpy=False)
                t2 = time.time()
                t3 = t2 - t1
                # print('time cost:', t3)
                total_time += t3
            # print("avg time:", total_time / total_num)
            # exit(0)
        '''
        # img = np.ones([1, 3, 320, 479]).astype('float32') 
        with fluid.scope_guard(self.detection_scope):
            detection, = self.exe.run(self.inference_program,
                              feed={self.feed_target_names[0]:img},
                              fetch_list=self.fetch_targets,
                              return_numpy=False)

            detection = np.array(detection)
            print("detection=", detection)
            print('\n')
        #print(detection.shape)
        #print('debug')
        #print(detection)
        # hzq
        '''
        with open('./test.pred', 'w') as f:
            f.write('%s\n%s\n'%(image_path, detection.shape[0]))
            for bbox_score in detection:
                xmin, ymin, xmax, ymax, score = bbox_score
                f.write('%s %s %s %s %s\n'%(xmin, ymin, xmax, ymax, score))
        '''
        if detection[0][0] == -1.:
        # if detection.shape == (1):
            print("No face detected")
            return np.array([[0, 0, 0, 0, 0]])

        det_conf = detection[:, 1]
        det_xmin = image_shape[2] * detection[:, 2] / shrink
        det_ymin = image_shape[1] * detection[:, 3] / shrink
        det_xmax = image_shape[2] * detection[:, 4] / shrink
        det_ymax = image_shape[1] * detection[:, 5] / shrink

        det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))
        #print('debug2')
        #print(det.shape)
        #print(det)
        return det


if __name__ == '__main__':
    # image_path = 'data/test.bmp'
    # image_path = "/Users/wangkeyao/Downloads/pic/177.1600264893626.png"
    image_path="../assets/images/demo.jpg"

    # detection = Detection(model_path='faceboxes_model_float32', use_gpu=False)
    # detection = Detection(model_path='blazeface_3x3_autodl', use_gpu=False)
    # detection = Detection(model_path='blazeface_0.1.0_int8', use_gpu=False)
    # detection = Detection(model_path='infer/blazeface_640_minsize128_model_final', model_filename = '__model__', params_filename = '__params__', use_gpu=False)
    # detection = Detection(model_path='faceboxes_blaze_small_float32_1001', model_filename = '__model__', params_filename = '__params__', use_gpu=False)
    # detection = Detection(model_path='blazeface_3x3_autodl', model_filename = 'model', params_filename = 'weights', use_gpu=False)
    # detection = Detection(model_path='blaze_smallest', model_filename = 'model', params_filename = 'params', use_gpu=False)
    # detection = Detection(model_path='blaze_fast', model_filename = 'model', params_filename = 'params', use_gpu=False)

    # detection = Detection(model_path='/Users/wangkeyao/Downloads/V1_int8-20191021/float/', model_filename = 'model', params_filename = 'weights', use_gpu=False)
    # detection = Detection(model_path='/Users/wangkeyao/Documents/rgb_face_detection_infer/float', model_filename = 'model', params_filename = 'weights', use_gpu=False)
    
    # detection = Detection(model_path='/Users/wangkeyao/Documents/output_noAssign_int8_for_AR/float', model_filename = 'model', params_filename = 'weights', use_gpu=False)
    # detection = Detection(model_path='/Users/wangkeyao/Downloads/V0.1.5-int8-0611/float_new', model_filename = 'model', params_filename = 'weights', use_gpu=False)

    # detection = Detection(model_path="/Users/wangkeyao/Downloads/detect_nir-customized-pa-faceid5_0.model.int8/float", model_filename = 'model', params_filename = 'weights', use_gpu=False)
    detection = Detection(model_path="../assets/models/detect_rgb-fp32", model_filename = '__model__', params_filename = '__params__', use_gpu=False)
    # detection = Detection(model_path="/Users/wangkeyao/Downloads/output_noAssign_fast_avg_AP_9896_recall_9901_int8/float", model_filename = 'model', params_filename = 'weights', use_gpu=False)
    # detection = Detection(model_path='/Users/wangkeyao/Documents/new', model_filename = 'model', params_filename = 'weights', use_gpu=False)
    # detection = Detection(model_path='/Users/wangkeyao/Downloads/paddle_rgb_detect', model_filename = 'model', params_filename = 'weights', use_gpu=False)
    # detection = Detection(model_path='/Users/wangkeyao/Downloads/haixiao_sdk/detection/output_noAssign_fast_avg_AP_9896_recall_9901_int8/float', model_filename = 'model', params_filename = 'weights', use_gpu=False)
    # detection = Detection(model_path="/Users/wangkeyao/Downloads/detect_rgb-customized-pa-kouzao.model.int8/float", model_filename = 'model', params_filename = 'weights', use_gpu=False)
    # detection = Detection(model_path="/Users/wangkeyao/Downloads/kouzhao_28/", model_filename = 'model', params_filename = 'weights', use_gpu=False)
    # detection = Detection(model_path="/Users/wangkeyao/Downloads/V0.1.5-int8-0611/float_new/", model_filename = 'model', params_filename = 'weights', use_gpu=False)
    '''
    cap = cv2.VideoCapture(0)
    while(1):
        ret, frame = cap.read()
        # show a frame
        # cv2.imshow("capture", frame)
        
        rect = detection.infer_one(frame)
        dect_conf_threshold = 0.30
        detection.draw_image(rect, frame, dect_conf_threshold)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow("capture", frame)
    '''
    # new_box = [593.9, 21.8, 833.16 ,353.55, 1]
    # old_box = [573.11, 39.23, 833.31, 365.47, 1]

    img = cv2.imread(image_path)
    print("img.shape=",img.shape)

    # img = Image.open(image_path)
    # img = img.resize((180, 320), Image.ANTIALIAS)


    total_time = 0
    total_num = 100

    # for i in range(100):
    #     t1 = time.time()
    print("img.shape[0]=",img.shape[0])
    scale = 320.0 / img.shape[0]
    # print scale, img.shape[1]
    # scale = 0.25
    # img = cv2.resize(img, (180, 320))

    # img = detection.load_image(image_path)
    # img = cv2.resize(img, (240, 320))
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    print("img.shape=",img.shape)
    rect = detection.infer_one(img)
    detection.draw_image(rect, img, 0.2)   # threshold
    # draw_img = detection.drawRotateRectange(img, rect, (255, 0 ,0), 2)
    # img = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))
    # cv2.imshow('demo.jpg', img)
    # cv2.waitKey(0)
        # t2 = time.time()
        # print('time cost:', t2 - t1)
        # total_time += t2 - t1

    # print("avg time:", total_time / total_num)
    # print rect
    rect = rect[0] / scale
    score = rect[-1]


    print('rect:', rect)
    print(rect[2] - rect[0], rect[3] - rect[1], (rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)
    print('score:', score)


