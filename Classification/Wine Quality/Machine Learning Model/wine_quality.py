## Importing Libraries.
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

## Importing Dataset.
dataset = pd.read_csv("dataset/winequality-white.csv", delimiter = ';')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

### Represents the separation to binary categories ('good' or 'bad' quality) or evaluation by score degree.
### If value is equal to '0', then will be made a score separation above 6.5 or less.
### From the other hand a precision score implementation will be adopt.
implementation = 0
if implementation == 0:
    temp_v = []
    for i in range(y.shape[0]):
        if y[i] > 6.5:
            temp_v.append(1)
        else:
            temp_v.append(0)
    y = np.array(temp_v)

## Split data to Training and Testing datasets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Define the Machine Learning models.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

## Hyperparameter Tuning function for RandomForectClassifier.
def rfc():
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10, 20]
    }
    grid_search = GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_grid, cv = 3, n_jobs = -1)
    grid_search.fit(X_train, y_train)
    best_rf_model = grid_search.best_estimator_
    y_pred_rf = best_rf_model.predict(X_test)
    
    return y_pred_rf

# Defines models and passes some basic parameters for each model.
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Neighbours Classifier": KNeighborsClassifier(n_neighbors = len(np.unique(y)), metric = 'minkowski', p = 2, n_jobs = -1),
    "Support Vector Classifier": SVC(C = 1, gamma = 0.1),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree Classifier": DecisionTreeClassifier(criterion = 'entropy'),
    "Random Forest Classifier": rfc()
}

## Define metrics to measure the efficieny of each model.
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Initializes an empty list to store results.
results = []
for name, model in models.items():
    if name != "Random Forest Classifier":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
         y_pred = model
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average = 'weighted')
    precision = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')
    
    results.append({
        "model": name,
        "accuracy": round(accuracy, 2),
        "f1-Score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    })
    
df_results = pd.DataFrame(results)

# Initializes dictionary in order to reveal the best model implementation.
best_model = {
    'model': '',
    'accuracy': 0.0
}

for index, row in df_results.iterrows():
    accuracy = row['accuracy']
    model = row['model']
    
    if best_model['accuracy'] < accuracy:
        best_model['model'] = model
        best_model['accuracy'] = accuracy
        
print(f"Best model is {best_model['model']} with accuracy {round(best_model['accuracy'] * 100, 2)}%")