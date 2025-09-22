import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


# Tuning function for cross validation by passing hyperparameters and estimator to extrapolate the best model.
def grid_search_cross_validation(estimator, param_grid, X_train, y_train):
    """
    Finds the best classification estimator after cross validation procedure.
    
    Args:
        estimator (object): Object of estimator's class.
        param_grid (dict): A dictionary of grid parameters for cross validation.
        X_train (array): Array of input values of observations.
        y_train (array): Array of outout/target values of observations.

    Returns:
        The best model (object) extrapolated after cross validation procedure ready for fit.
    """
    grid_search = GridSearchCV(estimator = estimator, 
                               param_grid = param_grid, 
                               cv = 4, 
                               n_jobs = -1)
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # The function returns the best model of cross validation procedure. 
    return grid_search.best_estimator_


# Calcualating the metrics to evaluate the succession of classifier.
def classfier_results(y_true, y_pred):
    """
    Calculates the results of classifier based on actual and predicted data targets.
    Args: 
        y_true (array): Array of actual data targets.
        y_pred (array): Array of predicted data.
    Returns:
        A list of following metrics: accuracy, f1-score, precision, recall.
    """
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred, average = 'weighted'), 4)
    precision = round(precision_score(y_true, y_pred, average = 'weighted'), 4)
    recall = round(recall_score(y_true, y_pred, average = 'weighted'), 4)

    return [accuracy, f1, precision, recall]


# A custom version of Confusion Matrix using custom function.
def custom_confusion_matrix(y_true, y_pred, title = "Confusion Matrix"):
    """
    A custom version of confusion metrix.

    Args:
        y_true (array): Array of actual targets.
        y_pred (array): Array of predicted targets.    
        title (str): Title of figure.
    """
    cf_matrix = confusion_matrix(y_true, y_pred)
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    
    ax = plt.axes()
    sns.heatmap(cf_matrix,
                annot = labels, 
                fmt = "", 
                cmap = "crest",
                linewidth = 1);
    ax.set_title(title);
