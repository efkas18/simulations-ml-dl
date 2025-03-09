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

## Splitting the dataset into the Training set and Test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Define the Machine Learning models.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import KFold

# Hyperparameter Tuning function for Random Forest
def rfc():
    param_grid = {
        'n_estimators': [100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)  # Fit GridSearch on training data
    return grid_search.best_estimator_

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=len(np.unique(y)), metric='minkowski', p=2, n_jobs=-1),
    "Support Vector Classifier": SVC(C=1, gamma=0.1),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree Classifier": DecisionTreeClassifier(criterion='entropy'),
    "Random Forest Classifier": rfc()  # Use the best model from GridSearchCV
}

# Initialize list to store results
results = []

# Training and evaluating each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    # Store results
    results.append({
        "model": name,
        "accuracy": round(accuracy, 2),
        "f1-Score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    })

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Find the best model based on accuracy
best_model = df_results.loc[df_results['accuracy'].idxmax()]
print(f"Best model is {best_model['model']} with accuracy {round(best_model['accuracy'] * 100, 2)}%")