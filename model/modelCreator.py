from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential

def modelCreator(num_classes):
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
	
	return model

def saveMNISTModel():
	
	modelWOweights = modelCreator(10).to_json()
	modelWOweights.save_weights('MNIST_weights.h5')
	with open("MNIST_model.json", "w") as json_file:
		json_file.write(modelWOweights)
	