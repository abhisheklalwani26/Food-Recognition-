import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
import joblib
import numpy as np
from sklearn.utils import shuffle


dataframe = pd.read_csv('dataset.csv')
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
# print(dataframe)

# Seperating the Labels and features
X = dataframe.drop(['label'], axis=1)
Y = dataframe["label"]

X_train, Y_train = X[0:1391], Y[0:1391]
X_test, Y_test = X[1391:], Y[1391:]

# checking if the labels and features are correctly marked
# print(X_train)
# print(Y_test)

# making model and saving it
model = svm.SVC(kernel='rbf', random_state=0, gamma=0.01, C=1)
model.fit(X_train, Y_train)
joblib.dump(model, "model1")
