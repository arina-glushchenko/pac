import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('file_1', type=str)
parser.add_argument('file_2', type=str)
parser.add_argument('P', type=float)
args = parser.parse_args()
file_1 = args.file_1
file_2 = args.file_2
P = args.P


with open(file_1, 'r') as f:
    array1 = np.array(list(map(int, f.read().split())))

with open(file_2, 'r') as f:
    array2 = np.array(list(map(int, f.read().split())))

if (len(array1) != len(array2)):
    print("Разный размер массивов")
    exit(0)

if (P < 0 or P > 1):
    print("Вероятность задана некорретно")
    exit(0)

mask = np.random.choice([0, 1], size=len(array1), p=[1 - P, P])

selected_array = np.where(mask, array2, array1)

print(selected_array)