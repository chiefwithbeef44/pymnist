import matplotlib.pyplot as plt
from getkey import getkey, keys


def predict_classes(model, x_train, x_test, x_imgID):
	for i in range(len(x_train)):
		pred = model.predict_classes(x=x_test[i].reshape(1, 28, 28, 1), verbose=1, batch_size=1)
		plt.imshow(x_imgID[i], cmap=plt.get_cmap('gray'))
		plt.show()
		if getkey.key == keys.RIGHT:
			plt.close()
		print pred