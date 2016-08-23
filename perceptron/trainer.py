import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from perceptron import Perceptron


def label(elem):
    elem = elem.tolist()
    return 1 if elem == 'Iris-setosa' else -1
def activate(input):
    return 1 if input >= 0 else -1
data_frame = pd.read_csv('./Iris.csv')
y = data_frame.iloc[0:100, 5].values

for element in np.nditer(y, flags=['refs_ok'],op_flags=['readwrite']):
    element[...] = label(element)

# extract sepal length and petal length
X = data_frame.iloc[0:100, [1, 3]].values
# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('./images/02_06.png', dpi=300)
#plt.show()

ppn = Perceptron(X, y)
ppn.train()
print(ppn.str())
plt.close()
print(ppn.errors_)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')

plt.tight_layout()
# plt.savefig('./perceptron_1.png', dpi=300)
plt.show()

# determine accuracy
ones = np.ones(100)
trial_x = np.c_[ones.T, X]
correct = 0
incorrect = 0
for training_sample, true_y in zip(trial_x, y):
    result = ppn.activator(ppn.net_input(training_sample)) - true_y
    if result == 0:
        correct += 1
    else:
        incorrect += 1

print(correct, incorrect)


from matplotlib.colors import ListedColormap

plt.close()
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('./perceptron_2.png', dpi=300)
plt.show()