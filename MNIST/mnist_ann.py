# MNIST - Modified National Institute of Standards and Technology

## Imports libraries.
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import tensorflow as tf

# print(tf.__version__)

## Imports training and testing datasets.
testing_dataset = pd.read_excel('datasets/testing_20k.xlsx', header = None)
training_dataset = pd.read_excel('datasets/training_10k.xlsx', header = None)

### Encoding y-target label data.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [784])], remainder = 'passthrough')
training_dataset = np.array(ct.fit_transform(training_dataset))
testing_dataset = np.array(ct.transform(testing_dataset))

### Splits datasets into input and target arrays.
X_train = np.array(training_dataset[:, 10:])
y_train = np.array(training_dataset[:, :10])
X_test = np.array(testing_dataset[:, 10:])
y_test = np.array(testing_dataset[:, :10])

## Part 2 - Building the ANN
### Initializing the ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 32, activation = 'relu'))
ann.add(tf.keras.layers.Dropout(0.3)) 
ann.add(tf.keras.layers.Dense(units = 10, activation = 'softmax'))

## Part 3 - Training the ANN
### Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, batch_size = 32, epochs = 25)

### Predicts the testing set results as probabilities.
y_pred = ann.predict(X_test)

## Testing Metrics
from sklearn.metrics import classification_report, accuracy_score
# Converts probabilities to binary values (0 or 1)
y_pred_conv = np.zeros_like(y_pred, dtype = float)
y_pred_conv[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
# Extrapolates testing accuracy.
test_accuracy = round(accuracy_score(y_test, y_pred_conv), 2) 
# Classification Report (includes Precision, Recall, F1-score)
class_report = classification_report(y_test, y_pred_conv)

print(f'Test Accuracy: {test_accuracy}')
print("\nClassification Report:")
print(class_report)
