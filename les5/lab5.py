import pandas as pd
import numpy as np

df = pd.read_csv('titanic_with_labels.csv', sep = ' ')

df['sex'] = df['sex'].map(lambda x: 1 if x == 'М' or x == 'м' or x == 'Мужчина' else (0 if x == 'Ж' or x == 'ж' else np.nan))

df['row_number'].fillna(df['row_number'].max(), inplace=True)

mean_liters_drunk = int(df[(df['liters_drunk'] >= 0) & (df['liters_drunk'] < 10)]['liters_drunk'].mean())

df.loc[(df['liters_drunk'] < 0) | (df['liters_drunk'] >= 10), 'liters_drunk'] = mean_liters_drunk

print(df.to_string())



