import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib.pyplot import figure
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from hmmlearn import hmm
import json


class UnsupervisedModel:

    def __init__(self):
        self.data_X = None
        self.data_X_clustered = None
        self.data_Y = None
        self.modelType = "HMM"
        self.model = None
        self.history = None
        self.clustering_model = None

    def cluster_data(self, n_clusters=30, print_confusion_matrix=False, data=None):
        if data is not None:
            considered_data = data
        else:
            considered_data = self.data_X
        matrice_raw = np.array(considered_data)
        matrice = np.fft.fft(matrice_raw[:, 150:800])[:, :55] # only low frequencies from [150:800] range
        matrice = np.concatenate((matrice.real, matrice.imag), axis=1)
        matrice = np.fft.fft(matrice)[:, :] # heuristic that improves the results fft on top of fft
        matrice = np.concatenate((matrice.real[:, :], matrice.imag[:, :]), axis=1)  # best :95,:95
        # to ameliorate the clustering will scale the fft obtained previously
        matrice = matrice.T
        scaler = MinMaxScaler((-1, 1))
        scaler.fit(matrice)
        matrice = scaler.transform(matrice)
        matrice = matrice.T
        # define the model
        self.clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
        # fit model and predict clusters
        if data is None:
            self.data_X_clustered = self.clustering_model.fit_predict(matrice)
        else:
            return self.clustering_model.fit_predict(matrice)
        if print_confusion_matrix:
            if self.data_Y is not None:
                letter_labels = list("abcdefghijklmnopqrstuvwxyz") + ["space"]
                true_labels = np.array([letter_labels.index(letter) if letter in letter_labels else 26
                                    for letter in self.data_Y])
                print(sklearn.metrics.confusion_matrix(true_labels, self.data_X_clustered))
            else:
                print("can't print confusion matrix because true labels (data_Y) are not set")

    def cluster_score(self):
        """
        :return: the cluster score the higher, the better
        """
        if self.data_Y is None:
            print("can't calculate score without true labels")
            return
        letter_labels = list("abcdefghijklmnopqrstuvwxyz") + ["space"]
        true_labels = np.array([letter_labels.index(letter) if letter in letter_labels else 26
                                for letter in self.data_Y])
        conf = sklearn.metrics.confusion_matrix(true_labels, self.data_X_clustered)
        somme = np.sum(conf)
        return (np.sum(conf.max(axis=0))) / somme

    def set_data(self, data, set_labels=True, shuffle=False):
        """
        populate train_data and test_data from data
        :param shuffle_test: boolean indicating whether or not to shuffle the test data
        :param split_factor: percentage of data that goes to test set , value between 0 and 1
        :param data: pandas DataFrame probably from Data.get_DataFrame
        :return: None
        """
        if set_labels:
            self.data_X = data.drop(len(data.iloc[0]) - 1, axis=1)
            self.data_Y = data.iloc[:, -1]
        else:
            self.data_X = data

        if shuffle:
            # random_state = 0 for reproducible results
            self.data_X, self.data_Y = shuffle(self.data_X, self.data_Y, random_state=0)

    def train_and_predict(self, num_iterations=50, text=True):
        if self.data_X_clustered is None:
            print("you need to cluster the data first")
            return
        V = np.array(self.data_X_clustered, dtype="int64")
        with open("../transition_matrix_french.json", "r") as write_file:
            a = json.load(write_file)
        a = np.array(a)
        a = (a.T / np.sum(a.T, axis=0)).T
        b = np.random.rand(27, 30)
        b = (b.T / np.sum(b.T, axis=0)).T
        initial_distribution = [1 / 27 if i == 26 else 1 / 27 for i in range(27)]
        initial_distribution = np.array(initial_distribution)
        self.model = hmm.MultinomialHMM(n_components=27, n_iter=num_iterations, init_params="", params="e", tol=0.01)
        self.model.startprob_ = initial_distribution
        self.model.transmat_ = a
        self.model.emissionprob_ = b

        self.model.fit([V])

        x = self.model.predict(V.reshape(-1, 1))
        if text:
            letters = list("abcdefghijklmnopqrstuvwxyz ")
            phrase = ""
            for i in x:
                phrase = phrase + letters[i]
            return phrase
        else:
            return x
        if self.data_Y is not None:
            print("character accuracy= ", self.hmm_accuracy(x, self.data_Y))

    def hmm_accuracy(self, y_pred, y_true, mode="single"):
        # y_pred is the predeted hidden states for our case will be an np.array(number of observations,)
        # of integers between 1 and 27
        # y_true is an np.array of strings with same shape
        # mode is either "single" : accuracy over each charater
        # or "pairs": accuracy over pairs of characters
        letters = list("abcdefghijklmnopqrstuvwxyz ")
        y_true = np.array([letters.index(letter) if letter in letters else 26 for letter in y_true])
        if mode == "single":
            ratio = 0
            for i in range(len(y_pred)):
                ratio += y_pred[i] == y_true[i]
            ratio = ratio / len(y_pred)
        elif mode == "pairs":
            ratio = 0
            for i in range(len(y_pred) - 1):
                ratio += y_pred[i] != y_true[i] and y_pred[i + 1] != y_true[i + 1]
            ratio = ratio / len(y_pred)
        elif mode == "words":
            n, m = self.words_accu(y_pred, y_true)
            ratio = m / n
        return ratio

    def words_accu(self, y_pred, y_true):
        values = np.array([26] + list(y_true) + [26])
        search_val = 26
        ii = np.where(values == search_val)[0]
        count = 0
        for i in range(len(ii) - 1):
            count += np.array_equal(y_pred[ii[i]:ii[i + 1]], y_true[ii[i]:ii[i + 1]])
        return len(ii) - 1, count

    def predict(self, data, text=True):
        V = np.array(self.cluster_data(data=data), dtype="int64")
        x = self.model.predict(V.reshape(-1, 1))
        if text:
            letters = list("abcdefghijklmnopqrstuvwxyz ")
            phrase = ""
            for i in x:
                phrase = phrase + letters[i]
            return phrase
        else:
            return x

