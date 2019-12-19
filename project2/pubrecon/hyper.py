from .model import RCNN

from keras.optimizers import Adam, Nadam
from keras.losses import categorical_crossentropy, logcosh

import talos
from talos.utils import lr_normalizer


class Hyper:
    def __init__(self, ImageData, params=None, seed=None, verbose=1):
        if params is None:
            raise ValueError("Error: `params` cannot be `null`.")

        self.ImageData = ImageData
        self.params = params
        self.seed = seed
        self.scan = None

    def optimize(self):
        x, y = self.ImageData.images, self.ImageData.labels

        def model(x_train, y_train, x_val, y_val, params):
            rcnn = RCNN(ImageData=self.ImageData, loss=self.params['loss'], opt=self.params['opt'],
                        lr=lr_normalizer(self.params['lr'], self.params['opt']), seed=self.seed,
                        verbose=0)

            return rcnn.train(epochs=self.params['epochs'], batch_size=self.params['batch_size'],
                             split_size=self.params['split_size'], checkpoint_path=None, early_stopping=False, verbose=0)

        self.scan = talos.Scan(x, y, params=self.params, model=model, experiment_name='rcnn', fraction_limit=.001)
        return self.scan


    # def get_best_params(self):








    # def __init__(self, ImageData, model_and_weights_path=None, seed=None, verbose=1, params=None):
    #     if params is None:
    #         raise ValueError("Error: Params cannot be null.")
    #     else:
    #         self.params = params
    #
    # def train_opti(self, x_train, y_train, x_val, y_val, params):
    #     super().__init__(ImageData=ImageData, model_and_weights_path=model_and_weights_path, loss=self.params['loss'],
    #                      opt=self.params['opt'], lr=lr_normalizer(self.params['lr'], self.params['optimizer']),
    #                      seed=seed, verbose=verbose)
    #     return super().train(epochs=self.params['epochs'], batch_size=self.params['batch_size'],
    #                          split_size=self.params['split_size'], checkpoint_path=None, early_stopping=True, verbose=1)
    #
    # def scan(self):
    #     return talos.Scan(x, y, params=self.params, model=self.train_opti, experiment_name='RCNN', fraction_limit=.001)
