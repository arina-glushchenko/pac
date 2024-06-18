from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

def evaluate_important_features(num_features):
    important_indices = feature_importances.argsort()[-num_features:][::-1]  # выбор нужного количества важных признаков
    X_val_important = X_val.iloc[:, important_indices]
    X_test_important = X_test.iloc[:, important_indices]

    rf_model.fit(X_val_important, y_val)
    rf_predict_important = rf_model.predict(X_test_important)
    rf_accuracy_important = accuracy_score(y_test, rf_predict_important)

    print(f'Точность модели Random Forest на {num_features} важных признаках:', rf_accuracy_important)


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


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
y_val = np.array(y_val)
y_test = np.array(y_test)


rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
rf_model.fit(X_val_scaled, y_val)
feature_importances = rf_model.feature_importances_


evaluate_important_features(8)
evaluate_important_features(4)
evaluate_important_features(2)