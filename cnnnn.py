
import glob
from numpy import savez_compressed
from numpy import asarray
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

train_data = list()
train_label = list()


def datacsv(label):
    # run this script for all labels
    dirList = glob.glob("dataset/train_set/"+label+"/*.jpg")

    for img_path in dirList:
        file_name = img_path.split("/")[2]
        print(file_name)
        image2 = load_img(img_path, target_size=(128, 128))
        #image2 = image2.resize((64, 64))
        image2 = img_to_array(image2, dtype='uint8')
        train_data.append(image2)

        # cv2.imshow("window",roi)
        tag = labels.index(label)
        train_label.append(tag)
        # train_label.append(label)


labels = ["Bhel Puri", "Burger", "Butter Chicken", "Chicken Lollipop",
          "Chocolate Bar", "Chocolate Cake", "Gulab Jamun",  "Palak Paneer", "Pizza"]

for j in range(9):
    datacsv(labels[j])


train_data = asarray(train_data, dtype='uint8')
train_label = asarray(train_label, dtype='uint8')
train_label = train_label.reshape(-1, 1)
print(train_data.shape)
print(train_label.shape)
savez_compressed('food_data.npz', train_data, train_label)
