import tensorflow as tf
import numpy as np
from neupy.algorithms import PNN
from numpy.fft import fft, ifft, rfft, rfftfreq, irfft, fftfreq
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from sklearn.metrics import classification_report, confusion_matrix
import os
from tensorflow.python.client import device_lib
from sklearn.utils import shuffle


class SupervisedModel:

    def __init__(self, model="ANN"):
        self.train_data_X = None
        self.train_data_Y = None
        self.test_data_X = None
        self.test_data_Y = None
        self.modelType = model
        self.model = None
        self.history = None
        self.encoder = LabelEncoder()

    def clean_data(self, dataFrame=None):
        alphabet = "azertyuiopmlkjhgfdsqnbvcxw" + "space"
        if dataFrame is None:
            # for training data set
            if self.train_data_Y is not None:
                indices_to_remove = []
                for i in range(len(self.train_data_Y)):
                    if str(self.train_data_Y.iloc[i]) not in alphabet:
                        indices_to_remove.append(i)
                for indice in indices_to_remove:
                    print("removing the following char cause it not in the considered alphabet ",
                          self.train_data_Y.iloc[indice])
                self.train_data_Y.drop(index=indices_to_remove, axis=0, inplace=True)
                self.train_data_X.drop(index=indices_to_remove, axis=0, inplace=True)
            else:
                print("can't clean empty train data")

            # for test data set
            if self.test_data_Y is not None:
                indices_to_remove = []
                for i in range(len(self.test_data_Y)):
                    if str(self.test_data_Y.iloc[i]) not in alphabet:
                        indices_to_remove.append(i)
                for indice in indices_to_remove:
                    print("removing the following char cause it not in the considered alphabet ",
                          self.test_data_Y.iloc[indice])
                self.test_data_Y.index = list(range(len(self.test_data_Y.index)))
                self.test_data_X.index = self.test_data_Y.index
                self.test_data_Y.drop(index=indices_to_remove, axis=0, inplace=True)
                self.test_data_X.drop(index=indices_to_remove, axis=0, inplace=True)
            else:
                print("can't clean empty test data")
        else:
            indices_to_remove = []
            for i in range(len(dataFrame)):
                if str(dataFrame.iloc[i, -1]) not in alphabet:
                    indices_to_remove.append(i)
            for indice in indices_to_remove:
                print("removing the following char cause it not in the considered alphabet ",
                      dataFrame.iloc[indice, -1])
            dataFrame.drop(index=indices_to_remove, axis=0, inplace=True)

    def set_train_test_data(self, data, split_factor=0.2, shuffle_test=True):
        """
        populate train_data and test_data from data
        :param shuffle_test: boolean indicating whether or not to shuffle the test data
        :param split_factor: percentage of data that goes to test set , value between 0 and 1
        :param data: pandas DataFrame probably from Data.get_DataFrame
        :return: None
        """
        data_X = data.drop(len(data.iloc[0]) - 1, axis=1)
        data_Y = data.iloc[:, -1]
        self.train_data_X, self.test_data_X, self.train_data_Y, self.test_data_Y = train_test_split(data_X, data_Y,
                                                                                                    test_size=split_factor,
                                                                                                    shuffle=shuffle_test)
        if not shuffle_test:
            # random_state = 0 for reproducible results
            self.train_data_X, self.train_data_Y = shuffle(self.train_data_X, self.train_data_Y, random_state=0)

        self.encoder.fit(self.train_data_Y)

    def set_test_data(self, data):
        self.test_data_X = data.drop(len(data.iloc[0]) - 1, axis=1)
        self.test_data_Y = data.iloc[:, -1]
        # we don't set the encoder since this function is supposed to be used only by loaded models

    def preprocess_data(self, data=None):
        scaler = MinMaxScaler()
        if data is None:
            if self.train_data_X is not None:
                self.train_data_X = pd.DataFrame((scaler.fit_transform(np.abs(rfft(self.train_data_X)).T)).T)
                self.train_data_X.columns = range(len(self.train_data_X.iloc[0]))
            else:
                print("couldn't preprocess train data cause it is equal to None")
            if self.test_data_X is not None:
                scaler = MinMaxScaler()
                self.test_data_X = pd.DataFrame((scaler.fit_transform(np.abs(rfft(self.test_data_X)).T)).T)
                self.test_data_X.columns = range(len(self.test_data_X.iloc[0]))
            else:
                print("couldn't preprocess test data cause it is equal to None")
        else:
            tempDataFrame = pd.DataFrame((scaler.fit_transform(np.abs(rfft(data)).T)).T)
            tempDataFrame.columns = range(len(tempDataFrame.iloc[0]))
            return tempDataFrame

    def initialize_model_from_data(self):
        if self.modelType == "PNN":
            std = 0.1
            self.model = PNN(std=std)
        elif self.modelType == "ANN":
            self.model = Sequential()
            self.model.add(layers.Dense(1024, activation='relu', input_shape=(len(self.train_data_X.iloc[0]),)))
            self.model.add(layers.Dense(512, activation='relu'))
            self.model.add(layers.Dense(512, activation='relu'))
            self.model.add(layers.Dense(len(np.unique(self.train_data_Y)), activation='softmax'))
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
        elif self.modelType == "CNN":
            self.model = Sequential()

            self.model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(len(self.train_data_X.iloc[0]), 1)))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Conv1D(32, 3, activation='relu'))
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(512, activation='relu'))
            self.model.add(layers.Dense(512, activation='relu'))
            self.model.add(layers.Dense(len(np.unique(self.train_data_Y)), activation='softmax'))
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')

    def train(self, removeCuda=True):
        """
        :param removeCuda: true if you have cuda installed and the GPU doesn't have enough memory for training.
        Try it with false first and if there is an error keep it True
        :return:
        """
        if self.modelType == "PNN":
            self.model.train()
        elif self.modelType == "ANN":
            if removeCuda:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            Y_train_transformed = self.encoder.transform(self.train_data_Y)
            EPOCHS = 50
            es_callback = tf.keras.callbacks.EarlyStopping(monitor='acc', patience=5,
                                                           restore_best_weights=True,
                                                           verbose=1)

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="acc",
                                                             factor=.3,
                                                             patience=2,
                                                             verbose=1,
                                                             min_delta=0.01)

            self.history = self.model.fit(self.train_data_X, Y_train_transformed,
                                          batch_size=6,
                                          callbacks=[es_callback, reduce_lr],
                                          # validation_split=0.1,
                                          epochs=EPOCHS,
                                          verbose=1)

        elif self.modelType == "CNN":
            Y_train_transformed = self.encoder.transform(self.train_data_Y)
            X_train_scaled_transformed = np.expand_dims(self.train_data_X, -1)
            EPOCHS = 20
            es_callback = tf.keras.callbacks.EarlyStopping(monitor='acc', patience=4,
                                                           restore_best_weights=True,
                                                           verbose=1)

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="acc",
                                                             factor=.3,
                                                             patience=2,
                                                             verbose=1,
                                                             min_delta=0.01)

            self.history = self.model.fit(X_train_scaled_transformed, Y_train_transformed,
                                          batch_size=12,
                                          callbacks=[es_callback, reduce_lr],
                                          # validation_split=0.1,
                                          epochs=EPOCHS,
                                          verbose=1)

    def predict_on_test_set(self, text=False):
        """
        :param text: boolean that decides to show the text or return raw prediction ie. probability vectors
        :return:
        """
        if self.modelType == "PNN":
            prediction = self.model.predict(self.test_data_X)
            print(classification_report(self.test_data_Y, prediction))
            return prediction
        elif self.modelType == "ANN":
            prediction = self.model.predict(self.test_data_X)
            prediction_labeled = self.encoder.inverse_transform([np.argmax(l) for l in prediction])
            print(classification_report(self.test_data_Y, prediction_labeled))
            if text:
                return prediction_labeled
            else:
                return prediction
        elif self.modelType == "CNN":
            prediction = self.model.predict(np.expand_dims(self.test_data_X, -1))
            prediction_labeled = self.encoder.inverse_transform([np.argmax(l) for l in prediction])
            print(classification_report(self.test_data_Y, prediction_labeled))
            if text:
                return prediction_labeled
            else:
                return prediction

    def predict(self, dataset, text=False):
        """
        same as predict_on_test_set but on dataset instead of test set
        :param dataset: list of list containing data vectors (X)
        :param text: boolean that decides to show the text or return raw prediction ie. probability vectors
        :return:
        """
        if self.modelType == "PNN":
            prediction = self.model.predict(dataset)
            return prediction
        elif self.modelType == "ANN":
            prediction = self.model.predict(dataset)
            if text:
                return self.encoder.inverse_transform([np.argmax(l) for l in prediction])
            else:
                return prediction
        elif self.modelType == "CNN":
            prediction = self.model.predict(np.expand_dims(dataset, -1))
            if text:
                return self.encoder.inverse_transform([np.argmax(l) for l in prediction])
            else:
                return prediction

    def save_model(self, model_name="model"):
        suffix = str(len(os.walk('../saved_models').next()[1]))
        self.model.save(f"saved_models/{model_name}_{suffix}")
        if self.encoder is not None:
            self.numpy.save(f"../saved_models/LabelEncoder_{suffix}.npy", self.encoder.classes_)

    def load_model(self, model_name="model_0", labelencoder_name=None, modelType="ANN"):
        self.model = tf.keras.models.load_model("../saved_models/" + model_name)
        self.modelType = modelType
        if labelencoder_name is not None:
            self.encoder.classes_ = np.load("../saved_models/" + labelencoder_name + ".npy", allow_pickle=True)
