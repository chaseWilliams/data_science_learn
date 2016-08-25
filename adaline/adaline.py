import numpy as np

class Perceptron:

    def __init__(self, values, true_output, epochs=10, learning_rate=.01):
        shape = np.shape(values)
        ones = np.ones(shape[0])
        self.x_ = np.c_[ones.T, values]
        self.y_ = true_output
        self.n_ = shape[0] # number of training samples
        self.m_ = shape[1] + 1 # number of features. +1 for x0, which is just ones
        self.epochs = epochs # number of passes over training dataset
        self.alpha = learning_rate # how fast it learns; avoid too-big numbers; they don't converge
        self.w_ = np.zeros((self.m_)) # the weights
        self.errors_ = [] # misclassifications in each epoch

    def net_input(self, x_vec):
        return self.w_.T.dot(x_vec)

    def activator(self, weighted_x):
        return 1 if weighted_x >= 0 else -1

    def train(self):
        for epoch in range(self.epochs):
            epoch_errors = 0
            for training_sample, true_y in zip(self.x_, self.y_):
                computed_y = self.activator(self.net_input(training_sample))
                weight_update = self.alpha * (true_y - computed_y) * training_sample.T
                self.w_ += weight_update
                for x in weight_update:
                    if x != 0:
                        epoch_errors += 1
            self.errors_.append(epoch_errors)
        return self


    def predict(self, x):
        output = []
        for elem in x:
            output.append(self.activator(self.net_input(elem)))
        return np.array(output)

    def str(self):
        print(str(self.n_) + "\n" +\
              str(self.m_) + "\n" +\
              str(self.x_) + "\n" +\
              str(self.y_) + "\n" +\
              str(self.w_))
