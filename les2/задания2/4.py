import string

dct = {'sad' : 'unhappy', 'big' : 'large', 'small' : 'tiny'}

s = input()
lst = s.split(' ')

for i in range(len(lst)):
    if lst[i].lower() in dct:
        if not lst[i].islower():
            lst[i] = dct[lst[i].lower()].capitalize()
        else:
            lst[i] = dct[lst[i]]

new_sentence = ' '.join(lst)
print(new_sentence)