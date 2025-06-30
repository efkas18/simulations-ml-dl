import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV


## Tuning function for cross validation by passing hyperparameters and estimator to extrapolate the best model.
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