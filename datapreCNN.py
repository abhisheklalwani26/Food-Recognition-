from PIL import Image
from matplotlib import pyplot
from numpy import np
image = Image.open('i1.jpg')
image = image.resize((64, 64))
data = []
image = np.array(image)
data.append(image)
image2 = Image.open('i2.jpg')
image2 = image2.resize((64, 64))
image2 = np.array(image2)
data.append(image2)
print(type(image2))
print(image2.shape)
print(type(data))
print(np.asarray(data).shape)
print(data[0])

pyplot.imshow(Image.fromarray(data[1]))
pyplot.show()
