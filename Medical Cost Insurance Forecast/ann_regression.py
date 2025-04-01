# Imports libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Loads dataset from csv.
dataset = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv")

# Converts categorical variable into dummy/indicator variables.
dataset_one_hot = pd.get_dummies(dataset, dtype = int)

# Separates features from targets.
X = dataset_one_hot.drop("charges", axis = 1)
y = dataset_one_hot["charges"]

# Creates training and testing datasets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Implements normalasation on feautuers data in order to has better understanding for those the ann.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Sets the global random seed.
tf.random.set_seed(0)

# Builds ANN for regression case study.
ann = tf.keras.Sequential()
ann.add(tf.keras.layers.Dense(units = 50, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 30, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 50, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 1))

# Compiles and trains ANN.
ann.compile(optimizer = 'adam', loss = 'mae', metrics = ['mae'])
history = ann.fit(X_train, y_train, epochs = 200, 
                  callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss', min_delta = 2, patience = 2)])

# Prints model's information.
print(f"\n{ann.summary()}")

# Plots history of training curve.
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")

# Validates ANN with testing dataset.
from sklearn.metrics import mean_absolute_percentage_error
y_pred = np.round(np.squeeze(ann.predict(X_test)).astype(np.float64), 2)

print(f"Testing Mean Absolute Percentage Error = {round(mean_absolute_percentage_error(np.round(y_test.to_numpy(), 2), y_pred), 2) * 100}%")
