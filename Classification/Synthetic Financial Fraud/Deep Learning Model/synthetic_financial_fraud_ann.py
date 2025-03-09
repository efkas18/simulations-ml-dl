import pandas as pd
import numpy as np
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

# print(tf.__version__)

dataset = pd.read_csv('../dataset/synthetic-financial-fraud.csv', delimiter = ',')
null_values_flag = dataset.isnull().values.any()
steps_vector = dataset['step'].unique()

X = dataset.iloc[:, [0, 1, 2, 4, 5, 7, 8]].values
y = dataset.iloc[:, -2].values

# print(dataset['type'].unique())

## Manipulation of raw data in order to fit the needs of ML models
### Encoding label data.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

### Splitting the dataset into the Training set and Test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Part 2 - Building the ANN
### Initializing the ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 64, activation='relu'))
ann.add(tf.keras.layers.Dense(units = 32, activation='relu'))
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


## Part 3 - Training the ANN
### Compiling the ANN
ann.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size = 2056, epochs = 50)


### Predicting the Test set results
y_pred = ann.predict(X_test)

### Testing Metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Convert probabilities to binary values (0 or 1)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
test_accuracy = round(accuracy_score(y_test, y_pred), 2) 
# Classification Report (includes Precision, Recall, F1-score)
class_report = classification_report(y_test, y_pred)

print(f'Test Accuracy: {test_accuracy}')
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(class_report)
