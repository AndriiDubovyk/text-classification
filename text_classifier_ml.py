"""A text classifier using TF-IDF vectorization and Support Vector Machine (SVM) for categorization.

This class provides a text classification system that utilizes TF-IDF vectorization for feature extraction
and Support Vector Machine (SVM) for categorization.

Attributes:
   model_save_path (str): The file path to save the trained models.
   vectorizer (TfidfVectorizer): TF-IDF vectorizer for feature extraction.
   classifier (SVC): Support Vector Machine classifier for categorization.
   categories (list of str): Predefined categories for classification.
   trained (bool): Flag indicating whether the classifier has been trained.
   preprocessor (TextPreprocessor): Text preprocessor for text preprocessing tasks.

Methods:
   train(categories, texts): Trains the classifier on the provided categories and texts.
   classify(texts): Classifies input texts into predefined categories.

Example:
   from text_classifier import TextClassifier

   classifier = TextClassifier(language='english', use_lemmatization=True)
   categories = ['business', 'science', 'sports']
   texts = ['An article about the latest stock market trends.',
            'A research paper on quantum computing.',
            'A review of the recent football match.']
   classifier.train(categories, texts)
   predicted_categories = classifier.classify(texts)
"""

import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from text_preprocessor import TextPreprocessor


class TextClassifier:
    def __init__(self, language='english', use_lemmatization=False, use_stemming=True):
        """Initialize the TextClassifier.

        Args:
            language (str, optional): The language used for text preprocessing. Defaults to 'english'.
            use_lemmatization (bool, optional): Whether to perform lemmatization during text preprocessing.
                Defaults to False.
            use_stemming (bool, optional): Whether to perform stemming during text preprocessing.
                Defaults to True.
        """
        self.model_save_path = "models/text_classifier_model.pkl"
        self.vectorizer = TfidfVectorizer()
        self.classifier = SVC(kernel='linear', probability=True)
        self.categories = []
        self.trained = False
        self.preprocessor = TextPreprocessor(language, use_lemmatization, use_stemming)

    def train(self, categories, texts):
        """Train the classifier on the provided categories and texts.

        Args:
            categories (list of str): Predefined categories for classification.
            texts (list of str): Text documents for training the classifier.
        """

        # Load the models if it exists
        if os.path.exists(self.model_save_path):
            with open(self.model_save_path, 'rb') as f:
                self.categories, self.vectorizer, self.classifier = pickle.load(f)

        # Update categories if new ones are provided
        for category in categories:
            if category not in self.categories:
                self.categories.append(category)

        # Train on preprocessed texts
        preprocessed_texts = [self.preprocessor.preprocess(text) for text in texts]
        x = self.vectorizer.fit_transform(preprocessed_texts)
        y = [self.categories.index(category) for category in categories]
        self.classifier.fit(x, y)
        self.trained = True

        # Save the models
        with open(self.model_save_path, 'wb') as f:
            pickle.dump((self.categories, self.vectorizer, self.classifier), f)

    def classify(self, texts):
        """Classify input texts into predefined categories.

        Args:
            texts (list of str): Text documents to be classified.

        Returns:
            list of str: Predicted categories for each input text.
        """
        if not self.trained:
            raise ValueError("Classifier has not been trained yet.")
        preprocessed_texts = [self.preprocessor.preprocess(text) for text in texts]
        x = self.vectorizer.transform(preprocessed_texts)
        predictions = self.classifier.predict(x)
        predicted_categories = [self.categories[prediction] for prediction in predictions]
        return predicted_categories
