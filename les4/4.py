import numpy as np

matrix = np.array([
    [3, 4, 5],
    [7, 10, 3],
    [2, 2, 2],
    [5, 12, 13],
    [6, 8, 10]
])


a, b, c = matrix.T
mask = (a + b > c) & (a + c > b) & (b + c > a)
print(matrix[mask], "- это длины сторон треугольника")




