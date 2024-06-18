import numpy as np

array = np.random.randint(10, size=10)

unique, counts = np.unique(array, return_counts=True)

ind = np.argsort(counts)[::-1]
sorted_by_freq1 = unique[ind]

sorted_by_freq2 = unique[np.argsort(counts)[::-1]]

print("Изначальный массив:", *array)
print("Отсортированный массив:", *sorted_by_freq2)