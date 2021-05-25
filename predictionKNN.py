from sklearn.preprocessing import StandardScaler
import time
import joblib
import cv2
import numpy as np
model = joblib.load("model")

# reading image and converting it into grayscale
im = cv2.imread("i1.jpg")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


roi = cv2.resize(im_gray, (64, 64), interpolation=cv2.INTER_AREA)

cv2.imwrite("predictKNN.jpg", roi)

rows, cols = roi.shape
sc = StandardScaler()


X = []

# #Add pixel into csv file
for i in range(rows):
    for j in range(cols):
        k = roi[i, j]
        X.append(k)
X = np.reshape(X, (64, 64))
X = sc.fit_transform(X)
X = np.reshape(X, 4096)
predictions = model.predict([X])
print("Prediction: ", predictions[0])

# cv2.waitKey(10000)
