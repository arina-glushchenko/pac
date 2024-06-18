import numpy as np
import pandas as pd


df = pd.read_csv('wells_info_na.csv')

for col in df.columns:
    if df[col].dtype == 'float64':
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].value_counts().idxmax(), inplace=True)

print(df.to_string())
