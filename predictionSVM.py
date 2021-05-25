import time
import joblib
import cv2
import numpy as np
model = joblib.load("model1")

# reading image and converting it into grayscale
im = cv2.imread("i5.jpg")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


roi = cv2.resize(im_gray, (64, 64), interpolation=cv2.INTER_AREA)

cv2.imwrite("predictSVM.jpg", roi)

rows, cols = roi.shape

X = []

# #Add pixel into csv file
for i in range(rows):
    for j in range(cols):
        k = roi[i, j]
        k = k/255
        X.append(k)


predictions = model.predict([X])
print("Prediction: ", predictions[0])
