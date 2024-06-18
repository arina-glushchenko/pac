import random

n = int(input())
lst = [random.randint(1, 1000) for i in range(n)]

odd = 0; even = 0
for i in lst:
    if (i%2 == 0):
        even += 1
    else:
        odd += 1
print("""Список: {}
Четных: {}
Нечетных: {}""".format(lst,even, odd))