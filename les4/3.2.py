import numpy as np

vector = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
window_size = 3
result = []

for i in range(len(vector)-window_size+1):
    result.append(0)
    for j in range(window_size):
        result[i] += vector[i+j]
    result[i] /= window_size
print("Исходный вектор:", vector)
print("Плавающее среднее:", result)