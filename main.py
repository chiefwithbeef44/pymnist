import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 200
image_height, image_width = 28, 28
num_classes = 10
epochs = 12

if __name__ == "__main__":

	#loads the data (downloads if necessary)
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	(x_img, y_img), (x_imgID, y_imgID) = mnist.load_data()

	#formats the data
	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, image_width, image_height)
		x_test = x_test.reshape(x_test.shape[0], 1, image_width, image_height)
		input_shape = (1, image_width, image_height)
	else:
		x_train = x_train.reshape(x_train.shape[0], image_width, image_height, 1)
		x_test = x_test.reshape(x_test.shape[0], image_width, image_height, 1)
		input_shape = (image_width, image_height, 1)


	#converts image into float32 values, either 0 or 1 for more convenient training.
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
					 activation='relu',
					 input_shape=input_shape))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer=keras.optimizers.Adadelta(),
				  metrics=['accuracy'])

	model.fit(x_train, y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  verbose=1,
			  validation_data=(x_test, y_test))
	
#	model.save(filepath="/home/Documents", overwrite=True,)
	
	for i in range(len(x_train)):
		pred = model.predict_classes(x=x_test[i].reshape(1, 28, 28, 1), verbose=1, batch_size=1)
		plt.imshow(x_imgID[i], cmap=plt.get_cmap('gray'))
		plt.show()
		print pred