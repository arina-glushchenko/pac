import cv2
import os
import random
import numpy as np

def generator(n):
    for i in range(n):
        augmentation = ['a', 'b', 'c', 'd']
        aug = random.choice(augmentation)
        aug = 'a'
        file = random.choice(files)
        path1 = "images/" + file
        path2 = "labels/" + file

        img = cv2.imread(path1)
        mask = cv2.imread(path2)
        result = img
        mask = mask.astype(img.dtype)
        cv2.imshow("img", img)
        h, w = img.shape[:2]
        match aug:
            case 'a':
                center = (w / 2, h / 2)
                angle = random.randint(0, 360)
                angle = 10
                print(angle)
                rot90 = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, rot90, (w, h))
                mask = cv2.warpAffine(mask, rot90, (w, h))
            case 'b':
                p = random.randint(0, 1)
                print(p)
                if p == 0:
                    img = img[::-1]
                    mask = mask[::-1]
                else:
                    img = cv2.flip(img, 1)
                    mask = cv2.flip(mask, 1)
            case 'c':
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)
                print(x, y, w, h)
                img = img[y:y + h, x:x + w]
                mask = mask[y:y + h, x:x + w]
            case 'd':
                img = cv2.GaussianBlur(img, (3, 3), 3)
                mask = cv2.GaussianBlur(mask, (3, 3), 3)

        files.remove(file)
        yield (img, mask)

# Example usage
n = 100
dir = "./labels"
files = os.listdir(dir)
generator = generator(n)
lst = []
for i in range(n):
    (img, mask) = next(generator)
    img = cv2.resize(img, (256, 128))
    mask = cv2.resize(mask, (256, 128))
    cv2.imshow("image", img)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    print(img)
    print(mask)