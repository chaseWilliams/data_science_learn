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

    def net_input(self, x_vec):
        return self.w_.T.dot(x_vec)

    def activator(self, weighted_x):
        return 1 if weighted_x >= 0 else -1

    def train(self):
        for epoch in range(self.epochs):
            print('epoch number ' + str(epoch))
            epoch_errors = 0
            for training_sample, true_y in zip(self.x_, self.y_):
                computed_y = self.activator(self.net_input(training_sample))
                weight_update = self.alpha * (true_y - computed_y) * training_sample.T
                print('before' + str(self.w_))
                print(weight_update)
                self.w_ += weight_update
                print('after' + str(self.w_))
                for x in weight_update:
                    if x != 0:
                        epoch_errors += 1
            self.errors_.append(epoch_errors)
        return self

    def predict(self, x):
        return self.activator(self.net_input(x))

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
learner.train()
print(learner.str())
#print('the dot product is')
#print(learner.net_input(np.array([ 1, 4, 5, 6])))