import numpy as np

class Perceptron:

    def __init__(self, values, true_output, epochs=10, learning_rate=.01):
        shape = np.shape(values)
        ones = np.ones(shape[0])
        print()
        self.x_ = np.c_[ones.T, values]
        self.y_ = true_output
        self.n_ = shape[0] # number of training samples
        self.m_ = shape[1] + 1 # number of features. +1 for x0, which is just ones
        self.epochs = epochs
        self.alpha = learning_rate
        self.w_ = np.zeros((self.m_))
        self.errors_ = []

    def net_input(self):
        return self.w_.T.dot( self.x_)

    def train(self):

        return self

    def str(self):
        print(str(self.n_) + "\n" +\
              str(self.m_) + "\n" +\
              str(self.x_) + "\n" +\
              str(self.y_) + "\n" +\
              str(self.w_))

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)
np.concatenate((a, b.T), axis=1)

X = np.array( [
    [1, 2, 3],
    [4, 5, 6]
])
y = np.array( [1, 4, 7] )

learner = Perceptron(X, y)
print(learner.str())

print('the dot product is')
print(learner.net_input())