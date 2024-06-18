import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.random (size=( 10 , 5 )))

df['mean'] = df[df > 0.3].mean(axis=1)

print(df)

