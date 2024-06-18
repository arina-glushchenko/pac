import pandas as pd
import numpy as np

df = pd.read_csv('wells_info.csv')


df['CompletionDate'] = pd.to_datetime(df['CompletionDate'])
df['PermitDate'] = pd.to_datetime(df['PermitDate'])

df['months_count'] = (df.CompletionDate - df.PermitDate) / np.timedelta64(1, 'D')//30

print(df)
