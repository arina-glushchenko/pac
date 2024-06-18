def read_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = list(map(int, line.strip().split()))
            matrix.append(row)
    return matrix


def multiply_matrices(matrix1, matrix2):
    rows1 = len(matrix1)
    cols1 = len(matrix1[0])
    rows2 = len(matrix2)
    cols2 = len(matrix2[0])

    if cols1 != rows2:
        raise ValueError(
            "Матрицы нельзя перемножить. Количество столбцов первой матрицы должно быть равно количеству строк второй матрицы.")

    result = []
    for i in range(rows1):
        row = []
        for j in range(cols2):
            element = 0
            for k in range(cols1):
                element += matrix1[i][k] * matrix2[k][j]
            row.append(element)
        result.append(row)
    return result


def write_matrix(file_path, matrix):
    with open(file_path, 'w') as file:
        for row in matrix:
            for element in row:
                file.write(str(element) + ' ')
            file.write('\n')


def main():
    input_file_path = input("Введите путь к файлу с матрицами: ")
    output_file_path = input("Введите путь к файлу с результатом: ")

    matrix1 = read_matrix(input_file_path)
    matrix2 = read_matrix(input_file_path)

    result = multiply_matrices(matrix1, matrix2)

    write_matrix(output_file_path, result)
    print("Результат успешно записан в файл")


if __name__ == "__main__":
    main()
