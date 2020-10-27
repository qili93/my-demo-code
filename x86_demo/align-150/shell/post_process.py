from __future__ import print_function

import numpy as np
import cv2

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

def postprocess():
    lite_out = np.fromfile("lite-out.raw", dtype=np.float32)
    lite_out.resize(1, 300)
    print("lite_out.shape=", lite_out.shape)
    print("lite_out.dtype=", lite_out.dtype)

    M = np.fromfile("face-mark.raw", dtype=np.float64)
    M.resize(2, 3)
    print("M.shape=", M.shape)
    print("M.shape=", M.dtype)

    lmk_pred = np.array(lite_out).reshape([-1,2])[:150, :]
    lmk_pred = affine_backward(lmk_pred, M)
    image = cv2.imread("../assets/images/face.jpg")
    image = draw_landmark(image, lmk_pred[:150, :])
    # print(lmk_pred[:150, :])
    cv2.imwrite("lite-out.jpg", image)

if __name__ == '__main__':
    postprocess()