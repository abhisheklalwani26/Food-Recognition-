from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from numpy import load
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from PIL import Image


data = load('food_data.npz')
X, y = data['arr_0'], data['arr_1']
# separate into train and test datasets
trainX, testX, trainY, testY = train_test_split(
    X, y, test_size=0.3, random_state=1)
# print(trainX.shape, trainY.shape, testX.shape, testY.shape)
pyplot.imshow(Image.fromarray(trainX[2]))
print(trainY[2])
pyplot.show()
myInt = 10
trainY = trainY.reshape(-1)
testY = testY.reshape(-1)
trainY = np_utils.to_categorical(trainY, num_classes=9)
testY = np_utils.to_categorical(testY, num_classes=9)
tx = len(trainX)
ty = len(testX)

#trainX = trainX.reshape(-1)
#testX = testX.reshape(-1)
#myInt1 = 255
#trainX[:] = [x / myInt1 for x in trainX]
#testX[:] = [x / myInt1 for x in testX]
#trainX = trainX.reshape(tx, 128, 128, 3)
#testX = testX.reshape(ty, 128, 128, 3)


# datagen = ImageDataGenerator(rescale=1.0/255.0)
# prepare iterators
# train_it = datagen.flow(trainX, trainY, batch_size=32)
# test_it = datagen.flow(testX, testY, batch_size=32)


cnn3 = Sequential()
cnn3.add(Conv2D(filters=64, kernel_size=3, padding='same',
                activation='relu', input_shape=(128, 128, 3)))
cnn3.add(MaxPooling2D((2, 2)))
cnn3.add(Dropout(0.25))

cnn3.add(Conv2D(64, kernel_size=3, activation='relu'))
cnn3.add(MaxPooling2D(pool_size=(2, 2)))
cnn3.add(Dropout(0.25))

cnn3.add(Conv2D(128, kernel_size=3, activation='relu'))
cnn3.add(Dropout(0.4))

cnn3.add(Flatten())

cnn3.add(Dense(128, activation='relu'))
cnn3.add(Dropout(0.3))
cnn3.add(Dense(9, activation='softmax'))

cnn3.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# cnn3.fit(train_it, steps_per_epoch=len(train_it),
# validation_data = test_it, validation_steps = len(test_it), epochs = 5, verbose = 0)
cnn3.fit(trainX, trainY, batch_size=32, epochs=50,
         validation_data=(testX, testY))

cnn3.save("modelcnn")
