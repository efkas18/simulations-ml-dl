import pandas as pd
import numpy as np
import tensorflow as tf

# print(tf.__version__)

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

