def progression(b, q):
    current = b
    while True:
        yield current
        current *= q

b = int(input("Введите b"))
q = int(input("Введите q"))

prog_gen = progression(b, q)

for i in range(10):
    print(next(prog_gen))