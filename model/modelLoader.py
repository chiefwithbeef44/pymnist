from keras.models import model_from_json


# noinspection PyPep8Naming
def loadFromJSON(json, weights):
	jsonFile = open(json+".json", "r")
	jsonModel = jsonFile.read()
	loadedFromJSON = model_from_json(jsonModel)
	finalModel = loadedFromJSON.load_weights(weights+".h5")
	finalModel.compile()
	return finalModel