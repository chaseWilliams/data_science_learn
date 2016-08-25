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
plt.show()
def train_perceptron(ppn):
    ppn.train()
    print(ppn.w_)
    plt.close()
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
    print('The percent correct is ' + str(float(correct) / float(correct + incorrect) * 100) + '%')

    plt.close()
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    ones = np.ones(xx.ravel().shape[0])
    Z = ppn.predict(np.c_[np.c_[ones.T,xx.ravel()],yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    ax.axis('off')

    # Plot also the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

    ax.set_title('Perceptron')
    plt.show()

train_perceptron(Perceptron(X, y))
