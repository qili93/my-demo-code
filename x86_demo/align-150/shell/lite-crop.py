from __future__ import print_function

import numpy as np
import cv2

IMAGE_FILE_PATH="../assets/images/face-crop.jpg"
OUTPUT_FILE_PATH = "../assets/images/lite-out.raw"
OUTPUT_SCORE_PATH = "../assets/images/lite-score.raw"

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

def lite_ouptut():    
    output = np.fromfile(OUTPUT_FILE_PATH, dtype=np.float32)
    output.resize(1, 300)
    print("output.shape=", output.shape)

    scores = np.fromfile(OUTPUT_SCORE_PATH, dtype=np.float32)
    scores.resize(1, 2)
    print("scores.shape=", scores.shape)
    print(scores)
    # print("output.shape=", output.dtype)

    lmk_pred = np.array(output).reshape([-1,2])[:150, :]
    print("lmk_pred.shape=", lmk_pred.shape)
    # print("lmk_pred.shape=", lmk_pred.dtype)
    # print(lmk_pred)

    out_sort = np.sort(lmk_pred, axis=None)
    print(out_sort)

    image = cv2.imread(IMAGE_FILE_PATH)
    image = draw_landmark(image, lmk_pred)
    cv2.imwrite("lite-out.jpg", image)

if __name__ == '__main__':
    lite_ouptut()