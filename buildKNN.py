from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier


dataframe = pd.read_csv('datasetKNN.csv')
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
# print(dataframe)
label = ["Bhel\nPuri", "Burger", "Butter\nChicken", "Chicken\nLollipop",
         "Chocolate\nBar", "Chocolate\nCake", "Gulab\nJamun",  "Palak\nPaneer", "Pizza"]
# Seperating the Labels and features
X = dataframe.drop(['label'], axis=1)
Y = dataframe["label"]

X_train, Y_train = X[0:1250], Y[0:1250]
X_test, Y_test = X[1250:], Y[1250:]
# feature scaling
st_x = StandardScaler()
X_train = st_x.fit_transform(X_train)
X_test = st_x.fit_transform(X_test)

# checking if the labels and features are correctly marked
# print(X_train)
# print(Y_test)


knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', p=1)
knn.fit(X_train, Y_train)
joblib.dump(knn, "model")
