import json
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.utils.document_parser import DocumentParser
from src.utils.text_preprocessor import TextPreprocessor


class TextClassifierNB:
    def __init__(self):
        self.model = None
        self.categories = None
        self.text_preprocessor = TextPreprocessor(language='english', use_stemming=False, use_lemmatization=True)

    def train(self, directory_path=None, preprocessed_text_df=None):
        if directory_path:
            df = DocumentParser.parse_files_to_df(directory_path)
            df['text'] = df['text'].apply(lambda txt: self.text_preprocessor.preprocess(txt))
        else:
            df = preprocessed_text_df.copy()

        # Get the categories
        self.categories = df['label'].astype('category').cat.categories.tolist()

        # Mapping text labels to indices
        label_to_index = {category: index for index, category in enumerate(df['label'].unique())}
        df['label'] = df['label'].map(label_to_index)

        # Split the dataset
        x_train, x_temp = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=25)
        labels_test_val = x_temp['label']
        x_test, x_val = train_test_split(x_temp, test_size=0.5, stratify=labels_test_val, random_state=25)

        # Define model
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('nb', MultinomialNB()),
        ])

        # Train model
        self.model.fit(x_train['text'], x_train['label'].astype('category').cat.codes)

        val_predictions = self.model.predict(x_val['text'])
        val_accuracy = accuracy_score(x_val['label'], val_predictions)

        test_predictions = self.model.predict(x_test['text'])
        test_accuracy = accuracy_score(x_test['label'], test_predictions)

        print("Validation accuracy: ", val_accuracy)
        print("Test accuracy: ", test_accuracy)

        return test_accuracy

    def save(self, directory):
        model_path = os.path.join(directory, "model.pkl")
        categories_path = os.path.join(directory, "categories.json")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(categories_path, 'w') as f:
            f.write(json.dumps(self.categories))

    def load(self, directory):
        model_path = os.path.join(directory, "model.pkl")
        categories_path = os.path.join(directory, "categories.json")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(categories_path, 'r') as f:
            self.categories = json.loads(f.read())

    def predict(self, user_text):
        processed_text = self.text_preprocessor.preprocess(user_text)
        predicted_probabilities = self.model.predict_proba([processed_text])[0]
        category_prob_tuples = [(i, prob) for i, prob in enumerate(predicted_probabilities)]
        # Sort the list of tuples by probability in descending order
        sorted_category_prob_tuples = sorted(category_prob_tuples, key=lambda x: x[1], reverse=True)
        results = {}
        for (category_index, probability) in sorted_category_prob_tuples:
            results[self.categories[category_index]] = probability
        return results
