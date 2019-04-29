import matplotlib.pyplot as plt
import time

def predict_classes(model, x_train, x_test, x_imgID):
	for i in range(len(x_train)):
		pred = model.predict_classes(x=x_test[i].reshape(1, 28, 28, 1), verbose=1, batch_size=1)
		plt.imshow(x_imgID[i], cmap=plt.get_cmap('gray'))
		plt.annotate(s=("prediction: "+str(pred)), xy=(0, 0), xytext=(-4, -2))
		plt.show(block=False)
		plt.pause(3)
		plt.close()