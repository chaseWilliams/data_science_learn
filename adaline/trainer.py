import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from adaline import AdalineGD

exit
def label(elem):
    elem = elem.tolist()
    return 1 if elem == 'Iris-setosa' else -1
def activate(input):
    return 1 if input >= 0 else -1
data_frame = pd.read_csv('../Iris.csv')
y = data_frame.iloc[0:100, 5].values

for element in np.nditer(y, flags=['refs_ok'],op_flags=['readwrite']):
    element[...] = label(element)

# extract sepal length and petal length
X = data_frame.iloc[0:100, [1, 3]].values


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.tight_layout()
# plt.savefig('./adaline_1.png', dpi=300)
plt.show()

# im not very smart