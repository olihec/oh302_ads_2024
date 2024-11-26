# This file contains code for suporting addressing questions in the data
from .config import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

def train_linear_model(X, y, test_size=0.2, random_state=42):
    """Trains a linear regression model using train_test_split.

    Args:
        X (array-like): The feature data.
        y (array-like): The target variable data.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.5.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to 42.

    Returns:
        tuple: A tuple containing the trained model and the test data (X_test, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model trained!")

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"R-squared: {r2}")
    print(f"Mean Squared Error: {mse}")
    return model