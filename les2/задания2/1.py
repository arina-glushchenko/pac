s = input()
if (s == s[::-1]):
    print("Строка {} - палиндром".format(s))
else:
    print("Строка {} не является палиндромом".format(s))