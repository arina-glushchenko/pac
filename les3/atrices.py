def read_matrix(file_name):
    matrix = []
    with open(file_name, 'r') as file:
        for line in file:
            row = list(map(int, line.strip().split()))
            matrix.append(row)
    return matrix


def sum_matrices(matrix1, matrix2):
    rows1 = len(matrix1)
    cols1 = len(matrix1[0])
    rows2 = len(matrix2)
    cols2 = len(matrix2[0])

    if rows1 != rows2 or cols1 != cols2:
        raise ValueError("Матрицы должны иметь одинаковое количество строк и столбцов.")

    result = []
    for i in range(rows1):
        row = []
        for j in range(cols1):
            element = matrix1[i][j] + matrix2[i][j]
            row.append(element)
        result.append(row)
    return result


file_name1 = input("Введите имя первого файла: ")
file_name2 = input("Введите имя второго файла: ")

matrix1 = read_matrix(file_name1)
matrix2 = read_matrix(file_name2)

result = sum_matrices(matrix1, matrix2)

print("Результат сложения матриц:")
for row in result:
    print(row)