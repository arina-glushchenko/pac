import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df = pd.read_csv("titanic_prepared.csv")

df["age_child"] = df["age_child"].map({True: 1, False: 0})
df["age_adult"] = df["age_adult"].map({True: 1, False: 0})
df["age_old"] = df["age_old"].map({True: 1, False: 0})
df["morning"] = df["morning"].map({True: 1, False: 0})
df["day"] = df["day"].map({True: 1, False: 0})
df["evening"] = df["evening"].map({True: 1, False: 0})
df = df.drop(df.columns[0], axis=1)
X = df.drop("label", axis =1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=1)

y_val = np.array(y_val)
y_test = np.array(y_test)
y_train = np.array(y_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
y_val = np.array(y_val)
y_test = np.array(y_test)


#Decision tree
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_scaled, y_train)
predict_dt = model_dt.predict(X_test_scaled)
accuracy_dt = accuracy_score(y_test, predict_dt)
print("Decision tree accuracy:", accuracy_dt)

#XGBoost
model_xgb = XGBClassifier()
model_xgb.fit(X_train_scaled, y_train)
predict_xgb = model_xgb.predict(X_test_scaled)
accuracy_xgb = accuracy_score(y_test, predict_xgb)
print("XGBoost accuracy:", accuracy_xgb)

#Logistic regression
model_lr = LogisticRegression()
model_lr.fit(X_train_scaled, y_train)
predict_lr = model_lr.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, predict_lr)
print("Logistic regression accuracy:", accuracy_lr)


feature_importances = model_dt.feature_importances_
important_indices = feature_importances.argsort()[-8:][::-1]
X_val_important = X_val.iloc[:, important_indices]
X_test_important = X_test.iloc[:, important_indices]


model_dt.fit(X_val_important, y_val)
predict_important = model_dt.predict(X_test_important)
accuracy_important = accuracy_score(y_test, predict_important)
print('Точность модели Decision tree на 2 важных признаках:',accuracy_important)








