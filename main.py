import keras
import predictor
import model.modelCreator as mC
from keras.datasets import mnist
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
	
	model = mC.createModel(10)
	mC.saveMNISTModel(model)
	mC.saveWeights(model)
	
	#predictor.predict_classes(model=model,x_test=x_test, x_train=x_train, x_imgID=x_imgID)