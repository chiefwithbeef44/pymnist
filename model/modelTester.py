# noinspection PyPep8Naming
def evaluateAccuracy(model, x_train, y_train, batch_size, epochs, x_test, y_test):
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
	return model.evaluate(x_test, y_test, verbose=0)
