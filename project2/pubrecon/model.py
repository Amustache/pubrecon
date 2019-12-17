import os
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from keras import Model
from keras import optimizers, losses
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


class RCNN:
    def __init__(self, ImageData, model_and_weights_path=None, loss=None, opt=None, lr=0.001, verbose=1):
        self.ImageData = ImageData
        if ImageData.get_num_samples() == 0 or ImageData.get_num_classes() == 0:
            print("Warning: ImagesData is empty.")
        self.lr = lr

        if loss is None:  # https://keras.io/losses/
            self.loss = losses.categorical_crossentropy
        else:
            self.loss = loss

        if opt is None:  # https://keras.io/optimizers/
            self.opt = optimizers.Adam(lr=self.lr)
        else:
            self.opt = opt

        self.hist = None
        self.classes = []

        cv2.setUseOptimized(True)

        if model_and_weights_path is None:
            # Import pretrained original VGG16 model with ImageNet weights
            vggmodel = VGG16(weights='imagenet', include_top=True)  # https://keras.io/applications/
            if verbose == 2:
                print(vggmodel.summary())  # Pretty sure I can optimize that thing...

            # Freeze first 15 layers
            for i, layers in enumerate(vggmodel.layers[:15]):
                layers.trainable = False
                if verbose == 2:
                    print("- Layer {} ({}) is not trainable anymore.".format(layers.get_config()['name'], i + 1))

            # Add a {number of classes} unit softmax dense layer
            predictions = Dense(self.ImageData.get_num_classes(), activation="softmax")(
                vggmodel.layers[-2].output)  # Maybe not all labels
            self.model = Model(input=vggmodel.input, output=predictions)

            # Compile the model using Adam optimizer with learning rate of 0.001 by default
            # We are using categorical_crossentropy as loss by default since the output of the model is categorical
            self.model.compile(loss=self.loss, optimizer=self.opt, metrics=["accuracy"])
        else:
            # Load model
            with open(os.path.join(model_and_weights_path, "model.json"), 'r') as json_model:
                self.model = model_from_json(json_model.read())
                json_model.close()

            # Load weights
            self.model.load_weights(os.path.join(model_and_weights_path, "weights.h5"))

            # Compile model
            self.model.compile(loss=self.loss, optimizer=self.opt, metrics=["accuracy"])

        if verbose:
            self.summary()

    def summary(self):
        print(self.model.summary())

    def save_model(self, model_and_weights_path):
        # Serialize and save model
        with open(os.path.join(model_and_weights_path, "model.json"), "w") as json_file:
            json_file.write(self.model.to_json())

        # Serialize and save weights
        self.model.save_weights(os.path.join(model_and_weights_path, "weights.h5"))

    def train(self, epochs=100, split_size=0.10, checkpoint_path=None, early_stopping=True, verbose=1):
        # One-hot encoding: Basically "unique-fy-ish" each class.
        # https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f
        class MyLabelBinarizer(LabelBinarizer):
            def transform(self, y):
                Y = super().transform(y)
                if self.y_type_ == 'binary':
                    return np.hstack((Y, 1 - Y))
                else:
                    return Y

            def inverse_transform(self, Y, threshold=None):
                if self.y_type_ == 'binary':
                    return super().inverse_transform(Y[:, 0], threshold)
                else:
                    return super().inverse_transform(Y, threshold)

        chosen_binarizer = MyLabelBinarizer()
        train_labels_fit = chosen_binarizer.fit_transform(self.ImageData.labels)
        self.classes = chosen_binarizer.classes_
        if verbose == 2:
            print("Classes: ", self.classes)

        # Test and train set, yay.
        X_train, X_test, y_train, y_test = train_test_split(self.ImageData.images, train_labels_fit, test_size=split_size)
        if verbose == 2:
            print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        # Dataset augmentation
        # This may not be needed following some magazines, as we do not often have rotated texts...
        # ... Or do we? Anyway it applies for the pictures so there's that.
        imgdatagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
        train_data = imgdatagen.flow(x=X_train, y=y_train)
        imgdatagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
        test_data = imgdatagen.flow(x=X_test, y=y_test)

        # We want checkpoints because losing training suckz lolz. https://keras.io/callbacks/#modelcheckpoint
        if checkpoint_path is not None:
            checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)
        else:
            checkpoint = None

        # If we are not doing any progress, stops the whole thing. https://keras.io/callbacks/#earlystopping
        if early_stopping:
            early = EarlyStopping(monitor='val_loss', min_delta=0, patience=epochs/10, verbose=1, mode='auto')
        else:
            early = None

        # FINALLY train the model. https://keras.io/models/sequential/#fit_generator
        steps = ceil(len(train_data) / len(chosen_binarizer.classes_))
        self.hist = self.model.fit_generator(generator=train_data,
                                             steps_per_epoch=steps, epochs=epochs, verbose=1, validation_data=test_data,
                                             validation_steps=steps, callbacks=[checkpoint, early])

    def predict(self, img):
        return self.model.predict(np.expand_dims(img, axis=0))

    def guess(self, img):
        predicts = self.predict(img)
        guess = self.classes[predicts.argmax()]
        proba = predicts.max()
        return guess, proba

    def history(self):
        return self.hist.history

    def test_image(self, id=None, path=None, show_infos=False, show_labels=False, number_of_results=2500, threshold=0.85, verbose=1):
        if id and path:
            raise ValueError("Error: Cannot have `id` and `path` at the same time.")
        elif id:
            sample_row = self.ImageData.files.iloc[id]
            sample_file = sample_row['file_name']
            sample_data = self.ImageData.df.loc[self.ImageData.df['file_name'] == sample_file]

            if show_infos:
                print(sample_data)

            if show_labels:
                self.ImageData.show_image(id, show_infos=False, show_labels=True)
        elif path:
            sample_file = path
        else:
            raise ValueError("Error: Must have `id` or `path`.")

        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        img = cv2.imread(sample_file)
        plt.figure(figsize=(18, 16))
        plt.subplot(1, 2, 1)
        plt.imshow(img)

        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        ss_results = ss.process()
        img_out = img.copy()

        for e, result in enumerate(tqdm(ss_results, disable=(verbose != 2))):
            if e < number_of_results:
                x, y, w, h = result
                timage = img_out[y:y + h, x:x + w]
                resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                img = np.expand_dims(resized, axis=0)
                out = self.model.predict(img)

                guess = self.classes[out.argmax()]
                guess_probability = out.max()

                if guess != "background":  # Eh... Like we should care.
                    if verbose == 2:
                        print("Found {} with probability {}.".format(guess, guess_probability))
                    elif verbose and guess_probability > threshold:
                        print("Found {} with probability {}.".format(guess, guess_probability))

                    if guess_probability > threshold:
                        cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(img, "{}".format(guess), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                    cv2.LINE_AA)

        plt.subplot(2, 2, 1)
        plt.imshow(img_out)
