# data_science_learn
Learning how to utilize data science libraries


# Tutorial

Learn how to go and start doing machine learning

#### Prerequisites:
* Familiarity with Python
* Knowledge of basic linear algebra (matrices)
    * Multivariate calculus is a definite plus, but not required
* Your own copy of [Python Machine Learning](https://www.amazon.com/Python-Machine-Learning-Sebastian-Raschka/dp/1783555130) (and check out the code at the [GitHub](https://github.com/rasbt/python-machine-learning-book))
* The [data](https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data)

Note that you can also clone this repository for the example code and the associated data, under the subfolder `haberman`.
### Configuring Environment

**Anaconda** is a package manager revered by the data science community that will significantly speed up and streamline your setup of your environment.

To install, go their [download](https://www.continuum.io/downloads) page. Download the *command line installer*, and `sh` it within terminal. The installation will occasionally prompt you for input, so be on standby

Great! Now you have Anaconda! In order to use it, you will generally follow the formula `conda [command]`. To install packages (such as python and many of the dependencies we'll need later on), you'll do `conda install [package-name]`. Anaconda supports multiple environments, and when you install a package it is automatically added to the root environment. However, if you want to add a new environment, simply do `conda create -n [name] [packages to incorporate, delimited with space]`. To switch environments, run `source activate [name]`. To return to root, run `source deactivate`. To list your environments, run `conda info --envs`

Note that to get help on the various conda subcommands, just running `conda` returns a list of them with short descriptions.

Now we need to get all the dependencies we'll need. Anaconda automatically installs some packages (like the Python version you chose), but we'll need to install others. Run `conda install` for
```
scikit-learn #great library with built in ML capabilities
numpy # highly efficient data manipulations allowing for vectorized code
pandas # we'll use this for reading the csv
matplotlib # open source data visualization tool, adapted from Matlab to Python
```

### Acquiring and Preprocessing the Data
**Pandas** lets us easily import csv files and it handles convenient translations to other data formats as well as great printing capabilities to observe your data. **Numpy** combines Python with lots of low-level C array actions, making code much more efficient and allowing for vectorized options, as opposed to looping through arrays. We'll need both of these libraries for handling the data. First we'll import our libraries:
```python
import pandas as pd
import numpy as np
```
Then we'll import the data using pandas' dataframe, its core object.
```python
df = pd.read_csv('./data.csv') # or any valid pathname to the file
X = df.iloc[:, :3]
Y = df.iloc[:, 3]
```
`iloc` is a method that essentially returns a `ndarray`, which is the core object of Numpy. The `[row, column]` syntax is how we denote what element within the array we want. The colon denotes that we want all, and when combined with a number, it means all up to or after the given number. So in this case, `[:, :3]` says that we want all the rows (training samples) and all the columns up until the third (starting from 0) column. If we were to run `print(df.tail())`, which prints the last five rows, we'll get a result like this:
```
      age  year  nodes  survival
301   75    62      1         1
302   76    67      0         1
303   77    65      3         1
304   78    65      1         2
305   83    58      2         2
```
Here the columns `age`, `year`, and `nodes` are our features, and `survival` is the target value.

Once we have the 'arrayed' version of our data, we'll want to split it into the features (data science jargon for traits, factors, etc.) and the target value (the *answer*). Our perceptron bases its predictions on the correlation between these *features* and the target values. However, we need a way to measure how accurate the perceptron turns out to be. Thus we'll have to split our data into *training* and *testing* datasets - one for the perceptron to fit, the other for the perceptron to predict. This is where `scikit-learn` comes in handy - it is essentially the swiss army knife of the data science world. We'll use a couple of functionalities from this library, so lets go ahead and import them:
```python
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
```
Now we have the `train_test_split` function imported, so we split the data.
```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
```
Note that test_size is where we determine what proportion of training/testing data we want. Here we state that we want 70% training and 30% testing.

Finally, before we train the perceptron, we need to scale the data within the range `[0, 1]` so that our perceptron fits it more efficiently- this is called **standardization**. Rather than computing, say, features that span from the ranges 1-10 and 100-10000, reducing them all allows for our computations to be less intensive (thank you Numpy). First we'll initialize our scaler
```python
scaler = StandardScaler()
```
and then we'll fit it
```python
scaler.fit(X_train)
```
followed by obtaining our standardized features
```python
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
```
Note that even though we call `transform` twice on separate data, it uses the exact same standardization factors because we fit it to our training data. Thus, we ensure that both the training and testing data is scaled down the same.
### Training with the Data
In order to fit the data, we'll use `scikit-learn`

### Visualize the Results


![picture][data_pic]

[data_pic]: https://raw.githubusercontent.com/chaseWilliams/data_science_learn/master/images/data_3d.png "Picture of Data"
