from keras.datasets import mnist
import matplotlib.pyplot as plt
image_width, image_height = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
for i in range(4):
	plt.subplot(221+i)
	plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))