# Loading the library with the iris dataset
from sklearn.datasets import load_iris
# Loading Random Forest Classifier Library
from sklearn.ensemble import RandomForestClassifier
# Loading Pandas
import pandas as pd
# Importing numpy
import numpy as np
# Setting random seed
np.random.seed(0)

# Creating an object called iris with iris dataset
iris  = load_iris()
# Create a dataframe with the four feature variables
df = pd.Dataframe(iris.data, columns = iris.feature_names)
df.head()

