import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import joblib
from sklearn.metrics import plot_confusion_matrix


dataframe = pd.read_csv('dataset.csv')
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
# print(dataframe)

# Seperating the Labels and features
X = dataframe.drop(['label'], axis=1)
Y = dataframe["label"]

X_train, Y_train = X[0:1391], Y[0:1391]
X_test, Y_test = X[1391:], Y[1391:]

label = ["Bhel\nPuri", "Burger", "Butter\nChicken", "Chicken\nLollipop",
         "Chocolate\nBar", "Chocolate\nCake", "Gulab\nJamun", "Jalebi", "Palak\nPaneer", "Pizza"]
# printing accuracy
model = joblib.load("model1")

predictions = model.predict(X_test)

print("Model Accuracy:", metrics.accuracy_score(Y_test, predictions))

fig, ax = plt.subplots(figsize=(35, 35))
disp = plot_confusion_matrix(model, X_test, Y_test,
                             display_labels=label,
                             cmap=plt.cm.Blues,
                             ax=ax)
disp.ax_.set_title("Confusion Matrix")


print(disp.confusion_matrix)

plt.show()
