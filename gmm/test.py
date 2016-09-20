import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture

data_frame = pd.read_csv('../Iris.csv')
y = data_frame.iloc[0:100, 5].values

# extract sepal length and petal length
X = data_frame.iloc[0:100, [1, 3]].values

def fit_samples(samples):
    gmm = mixture.GMM(n_components=2, covariance_type='full')
    gmm.fit(samples)
    print(type(gmm.score_samples(samples[0, :])))
    colors = ['r' if i==0 else 'g' for i in gmm.predict(samples)]
    ax = plt.gca()
    ax.scatter(samples[:,0], samples[:,1], c=colors, alpha=0.8)
    plt.show()

fit_samples(X)

#df = pd.read_csv()