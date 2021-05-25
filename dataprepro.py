import cv2
import os
import joblib
import numpy as np
import csv

import glob

header = ["label"]
for i in range(0, 4096):
    header.append("pixel"+str(i))
with open('dataset.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(header)


def datacsv(label):
    # run this script for all labels
    dirList = glob.glob("dataset/train_set/"+label+"/*.jpg")

    for img_path in dirList:
        file_name = img_path.split("/")[2]
        print(file_name)
        im = cv2.imread(img_path)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(im_gray, (64, 64), interpolation=cv2.INTER_AREA)

        # cv2.imshow("window",roi)

        data = []
        data.append(label)
        rows, cols = roi.shape

        # #Add pixel one-by-one into data Array.
        for i in range(rows):
            for j in range(cols):
                k = roi[i, j]
                k = k/255
                data.append(k)

        with open('dataset.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)

        # cv2.imwrite("proc_images/0/"+file_name, roi)

        cv2.waitKey()


label = ["Bhel Puri", "Burger", "Butter Chicken", "Chicken Lollipop",
         "Chocolate Bar", "Chocolate Cake", "Gulab Jamun", "Jalebi", "Palak Paneer", "Pizza"]

for j in range(10):
    datacsv(label[j])
