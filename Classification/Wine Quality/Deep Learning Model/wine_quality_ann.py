# Artificial Neural Network

## Importing the libraries.
import pandas as pd
import numpy as np
import tensorflow as tf

### The following line displays the version of TensorFlow library.
print(tf.__version__)

## Part 1 - Data Preprocessing
### Importing the dataset.
dataset = pd.read_csv("dataset/winequality-white.csv", delimiter = ';')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

## Encoding categorical data
### Represents the seperation to 'good' or 'bad' binary categories or evaluation by score.
### If value is equal to '0', then will be made a score seperation above 6.5 or less.
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
    loss = 'binary_crossentropy'
    output_units = 1
else:
    loss = 'categorical_crossentropy'
    output_units = len(np.unique(y))

    ### Change shape of target array, because need to be vertical and not horizontal 1d array for encoding.
    y = y.reshape(-1, 1)
    
    ### Encoding categorical data.
    from sklearn.preprocessing import OneHotEncoder
    ### Set sparse=False to get a NumPy array.
    encoder = OneHotEncoder(sparse_output = False)
    y = encoder.fit_transform(y)

### Splitting the dataset into the Training set and Test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

### Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Part 2 - Building the ANN
### Initializing the ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 128, activation='relu'))
ann.add(tf.keras.layers.Dense(units = 128, activation='relu'))
ann.add(tf.keras.layers.Dense(units = output_units, activation = 'sigmoid'))
#ann.add(tf.keras.layers.Dense(units = output_units, activation = 'softmax'))


## Part 3 - Training the ANN
### Compiling the ANN
ann.compile(optimizer='adam', loss = loss, metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

### Predicting the Test set results
y_pred = ann.predict(X_test)

### Testing Accuracy
from sklearn.metrics import accuracy_score
if implementation == 0:
    y_pred = (y_pred > 0.5)
    test_accuracy = round(accuracy_score(y_test, y_pred), 2)
else:
    y_pred_conv = np.zeros((y_pred.shape[0], y_pred.shape[1]), dtype = float)
    y_pred_conv[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
    test_accuracy = round(accuracy_score(y_test, y_pred_conv), 2)



