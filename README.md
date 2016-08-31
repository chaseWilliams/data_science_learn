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


### Configuring Environment

**Anaconda** is a package manager revered by the data science community that will significantly speed up and streamline your setup of your environment.

To install, go their [download](https://www.continuum.io/downloads) page. Download the *command line installer*, and `sh` it within terminal. The installation will occasionally prompt you for input, so be on standby

Great! Now you have Anaconda! In order to use it, you will generally follow the formula `conda [command]`. To install packages (such as python and many of the dependencies we'll need later on), you'll do `conda install [package-name]`. Anaconda supports multiple environments, and when you install a package it is automatically added to the root environment. However, if you want to add a new environment, simply do `conda create -n [name] [packages to incorporate, delimited with space]`. To switch environments, run `source activate [name]`. To return to root, run `source deactivate`. To list your environments, run `conda info --envs`*[]: 

Note that to get help on the various conda subcommands, just running `conda` returns a list of them with short descriptions.

Now we need to get all the dependencies we'll need. Anaconda automatically installs some packages (like the Python version you chose), but we'll need to install others. Run `conda install` for 
```
scikit-learn #great library with built in ML capabilities
numpy # highly efficient data manipulations allowing for vectorized code
pandas # we'll use this for reading the csv
matplotlib # open source data visualization tool, adapted from Matlab to Python
```

### Acquiring and Preprocessing the Data


### Training with the Data


### Visualize the Results


![picture][data_pic]

[data_pic]: https://raw.githubusercontent.com/chaseWilliams/data_science_learn/master/images/data_3d.png "Picture of Data"
