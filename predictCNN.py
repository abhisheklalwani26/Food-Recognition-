from tensorflow import keras
from keras.preprocessing.image import load_img
import numpy as np
from keras.preprocessing.image import img_to_array

labels = ["Bhel Puri", "Burger", "Butter Chicken", "Chicken Lollipop",
          "Chocolate Bar", "Chocolate Cake", "Gulab Jamun",  "Palak Paneer", "Pizza"]
model = keras.models.load_model('modelCNN')
image = load_img('i1.jpg', target_size=(128, 128))
image = img_to_array(image, dtype='uint8')
image = image.reshape(-1)
myInt1 = 255
image = image/255
#image[:] = [x / myInt1 for x in image]
image = image.reshape(128, 128, 3)
image = np.expand_dims(image, axis=0)
ypred = model.predict(image)
ypred = ypred.reshape(-1)
max1 = np.max(ypred)
iop = np.where(ypred == max1)
print(image.shape)
# print(ypred.shape)
print(ypred)
