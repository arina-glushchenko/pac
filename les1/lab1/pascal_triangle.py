import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('N', type=int)
args = parser.parse_args()
N = args.N

columns = []

for i in range(N):
    rows = []
    for j in range(i+1):
        if (j == 0 or i == j):
            rows.append(1)
        else:
            rows.append(columns[i - 1][j - 1] + columns[i - 1][j])
    if (i == N - 1):
        n_spaces = 0
    columns.append(rows)

n_spaces = N;

for i in range(N):
    for k in range(n_spaces):
        print(" " ,end="")
    for j in range(i+1):
        print(columns[i][j], end=" ")
    print()
    n_spaces-=1