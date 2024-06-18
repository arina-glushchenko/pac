import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = pd.read_csv("wells_info_with_prod.csv")

data["PermitDate"] = pd.to_datetime(data["PermitDate"])
data["PermitDate"] = (data["PermitDate"] - pd.to_datetime("2013-01-01")).dt.days

X = data[["PermitDate", "LatWGS84"]]
y = data["Prod1Year"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# transform train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
y_train_scaled = scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))

# transform test
X_test_scaled = scaler.fit_transform(X_test)
y_test_scaled = scaler.fit_transform(y_test.to_numpy().reshape(-1, 1))