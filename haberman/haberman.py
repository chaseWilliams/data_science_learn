import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

df = pd.read_csv('./data.csv')
print(df.tail())
X = df.iloc[:, :3]
y = df.iloc[:, 3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
ppn = Perceptron(n_iter=50, eta0=0.01, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
misclassified = (y_pred != y_test).sum()
print(type(misclassified))
print('Misclassified samples: %d' % misclassified)
integer = misclassified.item()
print('Perceptron accurateness: %d%%' % ((1 - float(misclassified.item()) / float(X.shape[0])) * 100))
