# Synthetic Financial Fraud Dataset

## Importing libraries.
import pandas as pd
import numpy as np

## Importing dataset.
dataset = pd.read_csv('../dataset/synthetic-financial-fraud.csv', delimiter = ',')
null_values_flag = dataset.isnull().values.any()
steps_vector = dataset['step'].unique()
X = dataset[dataset['step'].between(70, 100)].iloc[:, [1, 2, 4, 5]].values
y = dataset[dataset['step'].between(70, 100)].iloc[:, -2].values

## Manipulation of raw data in order to fit the needs of ML models
### Encoding label data.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

## Define the Machine Learning models.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

## Hyperparameter Tuning function for RandomForestClassifier.
def rfc():
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'n_estimators': [100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_grid, cv = 3, n_jobs = -1)
    
    return grid_search


# Defines models and passes some basic parameters for each model.
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Neighbours Classifier": KNeighborsClassifier(n_neighbors=len(np.unique(y)), metric='minkowski', p=2, n_jobs=-1),
    "Support Vector Classifier": SVC(C=1, gamma=0.1),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree Classifier": DecisionTreeClassifier(criterion='entropy'),
    "Random Forest Classifier": rfc()
}

# Initialize K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define metrics to measure the efficiency of each model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Initialize an empty list to store results
results = []

# K-Fold Cross-Validation for each model
for name, model in models.items():
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if name != "Random Forest Classifier":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            # Random Forest: Perform Grid Search and get the best model
            grid_search = model  # Already a GridSearchCV object
            grid_search.fit(X_train, y_train)  # Fit on training data
            best_rf_model = grid_search.best_estimator_  # Get the best model
            y_pred = best_rf_model.predict(X_test)  # Predict using the best model
        
        # Calculate metrics for each fold
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate the mean performance across all folds
    results.append({
        "model": name,
        "accuracy": round(np.mean(accuracies), 2),
        "f1-Score": round(np.mean(f1_scores), 2),
        "precision": round(np.mean(precisions), 2),
        "recall": round(np.mean(recalls), 2),
    })


df_results = pd.DataFrame(results)

# Initializes dictionary to reveal the best model implementation
best_model = {
    'model': '',
    'accuracy': 0.0
}

# Find the best model
for index, row in df_results.iterrows():
    accuracy = row['accuracy']
    model = row['model']
    
    if best_model['accuracy'] < accuracy:
        best_model['model'] = model
        best_model['accuracy'] = accuracy
        
print(f"Best model is {best_model['model']} with accuracy {round(best_model['accuracy'] * 100, 2)}%")
