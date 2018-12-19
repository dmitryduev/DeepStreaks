import os
import glob
from keras.models import load_model, model_from_json
import keras.backend as K


if __name__ == '__main__':

    path_models = '../models'

    models = glob.glob(os.path.join(path_models, '*.h5'))

    for model in models:
        print(f'{model}')
        K.clear_session()
        m = load_model(model)

        m.save_weights(model.replace('.h5', '.weights.h5'))
        model_json = m.to_json()
        with open(model.replace('.h5', '.architecture.json'), 'w') as json_file:
            json_file.write(model_json)
