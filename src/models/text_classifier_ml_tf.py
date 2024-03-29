import json

import joblib
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from src.models.text_preprocessor import TextPreprocessor


class TextClassifierTF:
    def __init__(self, language='english', use_lemmatization=False, use_stemming=True):
        self.model_path = 'models/tf/text_classifier_model.keras'
        self.vectorizer_path = 'models/tf/tfidf_vectorizer.pkl'
        self.label_names_path = 'models/tf/label_names.json'
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.label_binarizer = LabelBinarizer()
        self.model = None
        self.preprocessor = TextPreprocessor(language=language, use_lemmatization=use_lemmatization,
                                             use_stemming=use_stemming)
        self.label_names = None

    def train(self, X, Y, labels, epochs=10, batch_size=128, validation_split=0.1):
        X_preprocessed = [self.preprocessor.preprocess(text) for text in X]
        self.label_names = labels
        self.label_binarizer.fit(Y)
        num_categories = len(labels)

        Y_transformed = self.label_binarizer.transform(Y)

        # Split data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X_preprocessed, Y_transformed, test_size=0.2,
                                                            random_state=42)

        # Vectorize preprocessed text data
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)

        # Build the model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_categories, activation='softmax')
        ])

        # Compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Train the model
        self.model.fit(X_train_vectorized.toarray(), Y_train, epochs=epochs, batch_size=batch_size,
                       validation_split=validation_split)
        self.evaluate(X_test_vectorized.toarray(), Y_test)

    def save_model(self):
        if self.model is not None:
            self.model.save(self.model_path)
            print("Model saved successfully!")
        else:
            print("No model to save. Train a model first.")

        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, self.vectorizer_path)
            print("Vectorizer saved successfully!")
        else:
            print("No vectorizer to save. Train a model first.")

        if self.label_names is not None:
            with open(self.label_names_path, 'w') as f:
                json.dump(self.label_names, f)
            print("Label names saved successfully!")
        else:
            print("No label names to save. Train a model first.")

    def evaluate(self, X_test, Y_test):
        loss, accuracy = self.model.evaluate(X_test, Y_test)
        print("Test Accuracy:", accuracy)
        print("Test Loss:", loss)

    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model not found. Train a model first or provide correct model path.")

        try:
            self.vectorizer = joblib.load(self.vectorizer_path)
            print("Vectorizer loaded successfully!")
        except FileNotFoundError:
            print("Vectorizer not found. Train a model first or provide correct vectorizer path.")

        try:
            with open(self.label_names_path, 'r') as f:
                self.label_names = json.load(f)
            print("Label names loaded successfully!")
        except FileNotFoundError:
            print("Label names not found. Train a model first or provide correct label names path.")

    def predict_category(self, user_text):
        # Preprocess user input
        user_input_preprocessed = self.preprocessor.preprocess(user_text)
        user_input_vectorized = self.vectorizer.transform([user_input_preprocessed])

        # Predict category
        predicted_category_index = np.argmax(self.model.predict(user_input_vectorized), axis=-1)[0]
        return self.label_names[predicted_category_index]
