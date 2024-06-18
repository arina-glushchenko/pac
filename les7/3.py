import cv2
import os
import random

dir = "./labels"

files = os.listdir(dir)


path1 = "images/" + files[0]
img = cv2.imread(path1)

h, w = img.shape[:2]
center = (w / 2, h / 2)
angle = random.randint(0, 360)
print(center, h, w)
cv2.imshow("a", img)
img = cv2.flip(img, 1)
cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()