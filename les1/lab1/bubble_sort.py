import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('N', type=int)
args = parser.parse_args()
N = args.N

array = []

for i in range(0, N):
    array.append(random.random())

for i in range(N):
    for j in range(N - i - 1):
        if array[j] > array[j + 1]:
            array[j], array[j + 1] = array[j + 1], array[j]

for i in range(N):
    print(array[i])