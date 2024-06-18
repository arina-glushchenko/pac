s = input()
lst = s.split(' ')
maxlen = 0
maxword = ""
for i in lst:
    if (len(i) > maxlen):
        maxlen = len(i)
        maxword = i
print("Слово максимальной длины -", maxword)