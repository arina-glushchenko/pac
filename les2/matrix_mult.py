import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_name', type=str)
    parser.add_argument('output_file_name', type=str)
    args = parser.parse_args()
    input_file_name = args.input_file_name
    output_file_name = args.output_file_name
    read_matrix(input_file_name, output_file_name)

def read_matrix(input_file_name, output_file_name):
    matrix1 = []
    matrix2 = []
    with open(input_file_name, 'r') as f:
        flag = False
        for line in f:
            if line == "\n":
                flag = True
                continue
            if not flag:
                matrix1.append(line.split())
            else:
                matrix2.append(line.split())

    rows1 = len(matrix1)
    rows2 = len(matrix2)
    columns1 = len(matrix1[0])
    columns2 = len(matrix2[0])
    if columns1 != rows2:
        print("Размеры матриц некорретны")
        exit(0)

    result = count_matrix(rows1, rows2, columns1, columns2, matrix1, matrix2)
    print_result(result, output_file_name)

def count_matrix(rows1, rows2, columns1, columns2, matrix1, matrix2):
    result = []
    for i in range(rows1):
        row = []
        for j in range(columns2):
            el = 0
            for k in range(columns1):
                if matrix1[i][k].find('\n') != -1:
                    a = matrix1[i][k].find('\n')
                    matrix1[i][k] = matrix1[i][k][:a]

                if matrix2[k][j].find('\n') != -1:
                    a = matrix2[k][j].find('\n')
                    matrix2[k][j] = matrix2[k][j][:a]
                el += int(matrix1[i][k]) * int(matrix2[k][j])
            row.append(el)
        result.append(row)
    return result

def print_result(result, output_file_name):
    with open(output_file_name, 'w') as f:
        f.write("Result:\n")
        for row in result:
            f.write(' '.join(map(str, row)) + '\n')

main()