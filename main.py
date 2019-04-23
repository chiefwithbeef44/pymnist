from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils import np_utils
import numpy
from keras import backend as K

image_height, image_width = 28, 28

if __name__ == "__main__":

	#loads the data (downloads if necessary)
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	#plots the first four images in the dataset
	plt.subplot(221)
	plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
	plt.subplot(222)
	plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
	plt.subplot(223)
	plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
	plt.subplot(224)
	plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))
	plt.show()

	#formats the data
	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, image_width, image_height)
		x_test = x_test.reshape(x_test.shape[0], 1, image_width, image_height)
		input_shape = (1, image_width, image_height)
	else:
		x_train = x_train.reshape(x_train.shape[0], image_width, image_height, 1)
		x_test = x_test.reshape(x_test.shape[0], image_width, image_height, 1)
		input_shape = (image_width, image_height, 1)

	seed = 7
	numpy.random.seed(seed)

	numPix = x_train[1]*x_train[2]
	print(x_train.shape)
	x_train = x_train.reshape(x_train.shape[0], numPix).astype("float32")