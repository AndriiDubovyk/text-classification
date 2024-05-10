import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.utils.document_parser import DocumentParser
from src.utils.text_preprocessor import TextPreprocessor

VOCAB_SIZE = 10000
EMBEDDING_DIM = 64
BUFFER_SIZE = 10000
BATCH_SIZE = 64


class TextClassifierRNN:
    def __init__(self):
        self.model = None
        self.categories = None
        self.text_preprocessor = TextPreprocessor(language='english', use_stemming=False, use_lemmatization=True)

    def train(self, directory_path=None, preprocessed_text_df=None, epochs=50):
        start_time = time.time()
        if directory_path:
            df = DocumentParser.parse_files_to_df(directory_path)
            df['text'] = df['text'].apply(lambda txt: self.text_preprocessor.preprocess(txt))
        else:
            df = preprocessed_text_df.copy()

        # Get the categories
        self.categories = df['label'].astype('category').cat.categories.tolist()

        # Split the dataset
        x_train, x_temp = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=25)
        labels_test_val = x_temp['label']
        x_test, x_val = train_test_split(x_temp, test_size=0.5, stratify=labels_test_val, random_state=25)

        train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train['text'].to_numpy(), x_train['label'].astype('category').cat.codes))
        valid_ds = tf.data.Dataset.from_tensor_slices(
            (x_val['text'].to_numpy(), x_val['label'].astype('category').cat.codes))
        test_ds = tf.data.Dataset.from_tensor_slices(
            (x_test['text'].to_numpy(), x_test['label'].astype('category').cat.codes))

        train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        valid_ds = valid_ds.batch(BATCH_SIZE)
        test_ds = test_ds.batch(BATCH_SIZE)

        train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        valid_ds = valid_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
        encoder.adapt(train_ds.map(lambda text, label: text))

        # Define model
        self.model = tf.keras.Sequential([
            encoder,
            tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(self.categories), activation="softmax")
        ])
        self.model.summary()

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                          patience=2,
                                                          min_delta=1e-3,
                                                          verbose=1,
                                                          restore_best_weights=True)

        # Compile model
        self.model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                           optimizer=tf.keras.optimizers.RMSprop(),
                           metrics=['accuracy'])

        # Train model
        self.model.fit(train_ds,
                       validation_data=valid_ds,
                       callbacks=[early_stopping],
                       epochs=epochs)

        loss, acc = self.model.evaluate(test_ds)
        print(f"RNN training time: ${time.time() - start_time} seconds")
        print("RNN test accuracy: ", acc)
        print("RNN test loss: ", loss)
        return acc

    def save(self, directory):
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_path = os.path.join(directory, "model.keras")
        categories_path = os.path.join(directory, "categories.json")
        if self.model is not None:
            self.model.save(model_path)
        if self.categories is not None:
            with open(categories_path, 'w') as f:
                f.write(json.dumps(self.categories))

    def load(self, directory):
        model_path = os.path.join(directory, "model.keras")
        categories_path = os.path.join(directory, "categories.json")
        self.model = tf.keras.models.load_model(model_path)
        with open(categories_path, 'r') as f:
            self.categories = json.loads(f.read())

    def predict(self, user_text):
        processed_text = self.text_preprocessor.preprocess(user_text)
        predictions = self.model.predict(np.array([processed_text]))
        predicted_probabilities = predictions[0]
        category_prob_tuples = [(i, prob) for i, prob in enumerate(predicted_probabilities)]
        # Sort the list of tuples by probability in descending order
        sorted_category_prob_tuples = sorted(category_prob_tuples, key=lambda x: x[1], reverse=True)
        results = {}
        for (category_index, probability) in sorted_category_prob_tuples:
            results[self.categories[category_index]] = probability
        return results
