import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

dir = "./labels"

files = os.listdir(dir)


for file in files:
    path1 = "images/" + file
    path2 = "labels/" + file
    img = cv2.imread(path1)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(path2, 0)

    mask = mask.astype(img.dtype)
    _, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result = cv2.bitwise_and(img, img, mask=mask)
    result = cv2.drawContours(result, contours, -1, (252, 15, 192), 3)

    cv2.imshow("result", result)
    cv2.moveWindow("result", 100, 100)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
