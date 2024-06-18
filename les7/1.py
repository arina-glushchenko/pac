import cv2
import os
import random

def generator(n):
    for i in range(n):
        file = random.choice(files)
        path1 = "images/" + file
        path2 = "labels/" + file

        img = cv2.imread(path1)
        mask = cv2.imread(path2)
        lst.append(img)
        lst.append(mask)
        files.remove(file)
        yield lst

# Example usage
n = 2
dir = "./labels"
files = os.listdir(dir)
generator = generator(n)
lst = []
for i in range(n):
    result = next(generator)
    print(result)