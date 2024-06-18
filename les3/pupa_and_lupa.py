class pupa_and_lupa_parent:
    def __init__(self):
        self.salary_count = 0

    #Чтение и заполнение матрицы, проверка на размеры
    def read_matrix(self, file_name1, file_name2):
        matrix1 = []
        matrix2 = []
        with open(file_name1, 'r') as file:
            for line in file:
                matrix1.append(line.split())

        with open(file_name2, 'r') as file:
            for line in file:
                matrix2.append(line.split())

        rows1 = len(matrix1)
        cols1 = len(matrix1[0])
        rows2 = len(matrix2)
        cols2 = len(matrix2[0])

        if rows1 != rows2 or cols1 != cols2:
            print("Матрицы должны иметь одинаковое количество строк и столбцов.")

        return rows1, cols1, matrix1, matrix2

class Pupa(pupa_and_lupa_parent):
    #Получение зарплаты
    def take_salary(self, count):
        self.salary_count += count
        print("Зарплата Пупы теперь:", self.salary_count)

    #Сложение матрицы
    def do_work(self, filename1, filename2):
        rows, cols, matrix1, matrix2 = self.read_matrix(filename1, filename2)

        for i in range(rows):
            for j in range(cols):
                print(int(matrix1[i][j]) + int(matrix2[i][j]), end=" ")
            print()

class Lupa(pupa_and_lupa_parent):
    #Получение зарплаты
    def take_salary(self, count):
        self.salary_count += count
        print("Зарплата Лупы теперь:", self.salary_count)

    #Вычитание матрицы
    def do_work(self, filename1, filename2):
        rows, cols, matrix1, matrix2 = self.read_matrix(filename1, filename2)

        for i in range(rows):
            for j in range(cols):
                print(int(matrix1[i][j]) - int(matrix2[i][j]), end=" ")
            print()

class Accountant(Pupa, Lupa):
    def give_salary(self, worker, count):
        worker.take_salary(count)
