import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

df = pd.read_csv('./data.csv')
print(df.tail())
X = df.iloc[:, :3]
Y = df.iloc[:, 3]
# split X and Y values to training and testing groups
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
scaler = StandardScaler()

# standardize the data to a [0, 1) range, using same scale for both train and test data
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
ppn = Perceptron(n_iter=50, eta0=0.01, random_state=0)

# train the perceptron
ppn.fit(X_train_std, y_train)

# predict answers based on test data, tally misclassifications
y_pred = ppn.predict(X_test_std)
misclassified = (y_pred != y_test).sum()
print(type(misclassified))

print('Misclassified samples: %d' % misclassified)
integer = misclassified.item()
print('Perceptron accurateness: %d%%' % ((1 - float(misclassified.item()) / float(X.shape[0])) * 100))

# start visualizing the data points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = X.values
x_pos = np.empty((0,3), int)
x_neg = np.empty((0,3), int)

for sample, answer in zip(X, Y):
    if answer == 1:
        x_pos = np.vstack((x_pos, sample))
    else:
        x_neg = np.vstack((x_neg, sample))

x1 = x_pos[:, 0]
y1 = x_pos[:, 1]
z1 = x_pos[:, 2]
x2 = x_neg[:, 0]
y2 = x_neg[:, 1]
z2 = x_neg[:, 2]

ax.scatter(x1, y1, z1, c='r', marker='o')
ax.scatter(x2, y2, z2, c='b', marker='+')
ax.set_xlabel('Age')
ax.set_ylabel('Year (1900s)')
ax.set_zlabel('Nodes')

plt.show()