import random
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
        array1 = f.read()

with open(file_2, 'r') as f:
        array2 = f.read()

print(array1)
print(array2)

mask = np.random.choice([0, 1], size=len(array1), p=[1 - P, P])


selected_array = np.where(mask, array2, array1)

# Создание маски случайных выборов
mask = np.random.choice([0, 1], size=len(array1), p=[1 - P, P])

# Выбор элементов на основе маски
result = np.where(mask, array2, array1)

# Возвращение результатов

print(result)
