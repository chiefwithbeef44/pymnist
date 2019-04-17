from keras.datasets import mnist
import matplotlib
import keras

image_height, image_width = 28, 28

if __name__ == "__main__":

	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train.reshape(x_train.shape[0], 1, image_width, image_height)
	y_train = y_train.reshape(y_train.shape[0], 1, image_width, image_height)