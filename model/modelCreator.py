from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
import keras
import keras.backend as K
from keras.datasets import mnist

image_width, image_height = 28, 28
batch_size = 200
epochs = 12


# noinspection PyPep8Naming
def createModel(num_classes):
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	
	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, image_width, image_height)
		x_test = x_test.reshape(x_test.shape[0], 1, image_width, image_height)
		input_shape = (1, image_width, image_height)
	else:
		x_train = x_train.reshape(x_train.shape[0], image_width, image_height, 1)
		x_test = x_test.reshape(x_test.shape[0], image_width, image_height, 1)
		input_shape = (image_width, image_height, 1)
	
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	
	# converts image into float32 values, either 0 or 1 for more convenient training.
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	
	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
	
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
	
	return model


# noinspection PyPep8Naming
def saveMNISTModel(model):
	json_model = model.to_json()
	with open("MNIST_model.json", "w") as json_file:
		json = json_file.write(json_model)
	return json


# noinspection PyPep8Naming
def saveWeights(model):
	return model.save_weights("MNIST_weights.h5")