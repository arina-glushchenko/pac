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


df = pd.read_csv("train.csv")

df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

df["Embarked"].fillna("S", inplace=True)
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

mean_age = df["Age"].mean()
df["Age"].fillna(mean_age, inplace=True)


X = df[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]]
y = df["Survived"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1)
y_val = np.array(y_val)
y_test = np.array(y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'criterion': ['gini', 'entropy']
}

rf_model = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5)
grid_search_rf.fit(X_val_scaled, y_val)
best_rf_model = grid_search_rf.best_estimator_
rf_predict = best_rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_predict)
print('Точность модели Random Forest:',rf_accuracy)
print('Лучшие параметры Random Forest:',grid_search_rf.best_params_)

# XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15]
}

xgb_model = XGBClassifier()
grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5)
grid_search_xgb.fit(X_val_scaled, y_val)
best_xgb_model = grid_search_xgb.best_estimator_
xgb_predict = best_xgb_model.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, xgb_predict)
print('Точность модели XGBoost:',xgb_accuracy)
print('Лучшие параметры XGBoost:',grid_search_xgb.best_params_)

# Logistic Regression
param_grid_lr = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

lr_model = LogisticRegression()
grid_search_lr = GridSearchCV(lr_model, param_grid_lr, cv=5)
grid_search_lr.fit(X_val_scaled, y_val)
best_lr_model = grid_search_lr.best_estimator_
lr_predict = best_lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_predict)
print('Точность модели Logistic Regression:', lr_accuracy)
print('Лучшие параметры Logistic Regression:',grid_search_lr.best_params_)

# KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

knn_model = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn_model, param_grid_knn, cv=5)
grid_search_knn.fit(X_val_scaled, y_val)
best_knn_model = grid_search_knn.best_estimator_
knn_predict = best_knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_predict)
print('Точность модели KNN:',knn_accuracy)
print('Лучшие параметры KNN:',grid_search_knn.best_params_)