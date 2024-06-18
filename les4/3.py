import numpy as np

vector = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
window_size = 3

rolling_mean = np.convolve(vector, np.ones(window_size) / window_size, mode='valid')


print("Исходный вектор:", vector)
print("Плавающее среднее:", rolling_mean)