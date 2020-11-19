from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np
import cv2
import math

IMAGE_FILE_PATH="../assets/images/face.jpg"
MODEL_PATH="../assets/models/align150-fp32"

paddle.enable_static()

def standardize(img, mean, std):
    if img.shape[-1] == 1:
        img = np.repeat(img, axis=2)
    h, w, c = img.shape
    mean = np.array(mean).reshape([3,1])
    std = np.array(std).reshape([3,1])
    img = img.transpose([2,0,1]).reshape([c, -1])
    img = (img - mean) / std
    return img.reshape(c, h, w).transpose([1,2,0])

def prepocess(img, mean=[0.5,0.5,0.5], std=[1,1,1]):
    """
    mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    """
    h, w = img.shape[:2]
    img = img / 256.0
    img = standardize(img, mean, std)
    img = img.astype(np.float32).transpose([2,0,1])
    img = img.reshape([-1, 3, h, w])
    return img

def affine_backward(pts, M):
    coords = pts.copy()
    # import pdb; pdb.set_trace()
    for k, (px, py) in enumerate(pts):
        x = (M[1, 1] * (px - M[0, 2]) + M[0, 1] * (M[1, 2] - py)) / (M[0, 0] * M[1, 1] - M[1, 0] * M[0, 1])
        y = (M[1, 0] * (px - M[0, 2]) + M[0, 0] * (M[1, 2] - py)) / (M[0, 1] * M[1, 0] - M[1, 1] * M[0, 0])
        coords[k, 0], coords[k, 1] = x, y
    return coords

def draw_landmark(img, pts, dsize = 1, color = (0, 0, 255), number = False):
    '''Example function with types documented in the docstring.'''
    if len(pts) == 0:
        print("No point found !")
    else:
        for j in range(pts.shape[0]):
            location = (int(pts[j, 0]), int(pts[j, 1]))
            cv2.circle(img, location, dsize, color, -1)
            if number:
                location = (location[0] + 10, location[1] + 5)
                cv2.putText(img, str(j), location, cv2.FONT_HERSHEY_COMPLEX, 0.4, (255,0,0), 2)
    return img

def crop(image, pts=np.array([[141, 468], [409, 468], [409, 736], [141, 736]]), shift=0, scale=1.3, rotate=0, res=(128, 128)):
    """checked"""
    if pts.shape[0] == 150 or pts.shape[0] == 72:
        idx1 = 13 # 26 if pts_transform == True else 13
        idx2 = 34 # 24 if pts_transform == True else 34
    elif pts.shape[0] == 4:
        idx1 = 0
        idx2 = 1
    else:
        print("==> Error: abnormal length of pts !")

    # angle between the eyes
    alpha = 0
    if pts[idx2, 0] != -1 and pts[idx2, 1] != -1 and pts[idx1, 0] != -1 and pts[idx1, 1] != -1:
        alpha = math.atan2(pts[idx2, 1] - pts[idx1, 1], pts[idx2, 0] - pts[idx1, 0]) * 180 / math.pi
      
    # pts[pts == -1] = np.inf
    coord_min = np.min(pts, 0)
    # pts[pts == np.inf] = -1
    coord_max = np.max(pts, 0)
    
    # coordinates of center point
    c = np.array([coord_max[0] - (coord_max[0] - coord_min[0]) / 2, 
        coord_max[1] - (coord_max[1] - coord_min[1]) / 2])  # center
    max_wh = max((coord_max[0] - coord_min[0]) / 2, (coord_max[1] - coord_min[1]) / 2)

    # Shift the center point, rot add eyes angle
    c = c + shift * max_wh
    rotate = rotate + alpha

    M = cv2.getRotationMatrix2D((c[0], c[1]), rotate, res[0] / (2 * max_wh * scale))
    M[0, 2] = M[0, 2] - (c[0] - res[0] / 2.0)
    M[1, 2] = M[1, 2] - (c[1] - res[1] / 2.0)

    print("max_wh=", max_wh)
    print("scale=", scale)

    image_mini = cv2.warpAffine(image, M, res)
    cv2.imwrite("face-crop.jpg", image_mini)
    return image_mini, M

def read_image(image_path=IMAGE_FILE_PATH):
    image = cv2.imread(image_path)
    img_mini, M = crop(image)
    img_np = prepocess(img_mini)
    print("img_np.shape=", img_np.shape)
    print("img_np.shape=", img_np.dtype)
    print("M.shape=", M.shape)
    print("M.shape=", M.dtype)
    with open("face-input.raw", "wb") as f:
        img_np.tofile(f)
    with open("face-mark.raw", "wb") as f:
        M.tofile(f)
    return img_np, M

def read_raw_file():
    img_np = np.fromfile("face-input.raw", dtype=np.float32)
    img_np.resize(1, 3, 128, 128)
    print("img_np.shape=", img_np.shape)
    print("img_np.shape=", img_np.dtype)
    M = np.fromfile("face-mark.raw", dtype=np.float64)
    M.resize(2, 3)
    print("M.shape=", M.shape)
    print("M.shape=", M.dtype)
    return img_np, M

def infer_model(model_path=MODEL_PATH, image_path=IMAGE_FILE_PATH):
    if model_path is None:
        return

    # img_np = np.ones([1, 3, 128, 128]).astype('float32')
    # img_np, M = read_image(IMAGE_FILE_PATH)
    img_np, M = read_raw_file()
    
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.executor.global_scope()
    with fluid.scope_guard(inference_scope):
      [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
                                                 model_path, exe, '__model__', '__params__')
      landmark_out, face_score = exe.run(inference_program,
                    feed={feed_target_names[0]: img_np},
                    fetch_list=fetch_targets,
                    return_numpy=False)

    output = np.array(landmark_out)
    with open("infer-out.raw", "wb") as f:
        output.tofile(f)

    lmk_pred = np.array(landmark_out).reshape([-1,2])[:150, :]
    lmk_pred = affine_backward(lmk_pred, M)
    image = cv2.imread(image_path)
    image = draw_landmark(image, lmk_pred[:150, :])
    # print(lmk_pred[:150, :])
    cv2.imwrite("infer-out.jpg", image)

if __name__ == '__main__':
    infer_model(model_path='../assets/models/align150-fp32')