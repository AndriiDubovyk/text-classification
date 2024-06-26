import os

from src.classifiers.text_classifier_nb import TextClassifierNB
from src.classifiers.text_classifier_rnn import TextClassifierRNN
from src.classifiers.text_classifier_svm import TextClassifierSVM


class TextClassifierComplex:
    def __init__(self):
        self.categories = None
        self.classifer_nb = None
        self.classifer_svm = None
        self.classifer_rnn = None

    def train(self, directory_path=None, preprocessed_text_df=None, use_nb=True, use_svm=True, use_rnn=True):
        accuracy_values = []
        self.classifer_nb = None
        self.classifer_svm = None
        self.classifer_rnn = None
        if use_nb:
            self.classifer_nb = TextClassifierNB()
            accuracy = self.classifer_nb.train(directory_path=directory_path, preprocessed_text_df=preprocessed_text_df)
            accuracy_values.append(accuracy)
            self.categories = self.classifer_nb.categories
        if use_svm:
            self.classifer_svm = TextClassifierSVM()
            accuracy = self.classifer_svm.train(directory_path=directory_path,
                                                preprocessed_text_df=preprocessed_text_df)
            accuracy_values.append(accuracy)
            self.categories = self.classifer_svm.categories
        if use_rnn:
            self.classifer_rnn = TextClassifierRNN()
            accuracy = self.classifer_rnn.train(directory_path=directory_path,
                                                preprocessed_text_df=preprocessed_text_df)
            accuracy_values.append(accuracy)
            self.categories = self.classifer_rnn.categories
        return sum(accuracy_values) / len(accuracy_values)

    def save(self, directory):
        if self.classifer_nb:
            self.classifer_nb.save(os.path.join(directory, "nb"))
        if self.classifer_svm:
            self.classifer_svm.save(os.path.join(directory, "svm"))
        if self.classifer_rnn:
            self.classifer_rnn.save(os.path.join(directory, "rnn"))

    def load(self, directory):
        nb_path = os.path.join(directory, "nb")
        svm_path = os.path.join(directory, "svm")
        rnn_path = os.path.join(directory, "rnn")
        if os.path.exists(nb_path):
            self.classifer_nb = TextClassifierNB()
            self.classifer_nb.load(nb_path)
            self.categories = self.classifer_nb.categories
        if os.path.exists(svm_path):
            self.classifer_svm = TextClassifierSVM()
            self.classifer_svm.load(svm_path)
            self.categories = self.classifer_svm.categories
        if os.path.exists(rnn_path):
            self.classifer_rnn = TextClassifierRNN()
            self.classifer_rnn.load(rnn_path)
            self.categories = self.classifer_rnn.categories

    def predict(self, user_text, nb_weight=1, svm_weight=1, rnn_weight=1):
        results = {}
        sum_weights = nb_weight + svm_weight + rnn_weight
        if self.classifer_nb:
            nb_prediction = self.classifer_nb.predict(user_text)
            for category, probability in nb_prediction.items():
                results[category] = results.get(category, 0) + probability * nb_weight / sum_weights

        if self.classifer_svm:
            svm_prediction = self.classifer_svm.predict(user_text)
            for category, probability in svm_prediction.items():
                results[category] = results.get(category, 0) + probability * svm_weight / sum_weights

        if self.classifer_rnn:
            rnn_prediction = self.classifer_rnn.predict(user_text)
            for category, probability in rnn_prediction.items():
                results[category] = results.get(category, 0) + probability * rnn_weight / sum_weights

        sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
        rounded_results = {key: round(value, 2) for key, value in sorted_results.items()}
        return rounded_results
