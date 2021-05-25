import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import joblib
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler

dataframe = pd.read_csv('datasetKNN.csv')
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
# print(dataframe)

# Seperating the Labels and features
X = dataframe.drop(['label'], axis=1)
Y = dataframe["label"]

X_train, Y_train = X[0:1250], Y[0:1250]
X_test, Y_test = X[1250:], Y[1250:]

st_x = StandardScaler()
X_train = st_x.fit_transform(X_train)
X_test = st_x.fit_transform(X_test)

label = ["Bhel\nPuri", "Burger", "Butter\nChicken", "Chicken\nLollipop",
         "Chocolate\nBar", "Chocolate\nCake", "Gulab\nJamun", "Palak\nPaneer", "Pizza"]
# printing accuracy
knn = joblib.load("model")

predictions = knn.predict(X_test)
print("Model Accuracy:", metrics.accuracy_score(Y_test, predictions))

fig, ax = plt.subplots(figsize=(35, 35))
disp = plot_confusion_matrix(knn, X_test, Y_test,
                             display_labels=label,
                             cmap=plt.cm.Blues,
                             ax=ax)
disp.ax_.set_title("Confusion Matrix")


print(disp.confusion_matrix)

plt.show()
