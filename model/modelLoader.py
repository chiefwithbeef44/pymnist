from keras.models import model_from_json
import keras


# noinspection PyPep8Naming
def loadFromJSON(json, weights):
	json_file = open(json, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(weights)
	return loaded_model