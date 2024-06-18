clines = 0; cwords = 0; csymbols = 0

with open('input.txt', 'r') as f:
    for line in f:
        clines += 1
        cwords += (len(line.split()))
        csymbols += len(line.replace('\n', ''))
print("""Кол-во строк: {}
Кол-во слов: {}
Кол-во символов: {}""".format(clines, cwords, csymbols))
