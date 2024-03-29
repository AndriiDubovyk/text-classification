"""A class for preprocessing text data with options for lemmatization and stemming.

This module provides a TextPreprocessor class that preprocesses text data by tokenizing,
removing stopwords, and optionally performing lemmatization and stemming.

Attributes:
    language (str): The language of the text data, default is 'english'.
    use_lemmatization (bool): Whether to perform lemmatization, default is False.
    use_stemming (bool): Whether to perform stemming, default is True.

Methods:
    preprocess(text): Preprocesses the input text by tokenizing, removing stopwords,
        and optionally performing lemmatization and stemming.

Example:
    from text_preprocessor import TextPreprocessor

    preprocessor = TextPreprocessor(language='english', use_lemmatization=True)
    preprocessed_text = preprocessor.preprocess("This is an example sentence.")
"""

import string

import nltk
import pandas
import pymorphy2  # pip install pymorphy2-dicts-uk
import uk_stemmer  # pip install git+https://github.com/Desklop/Uk_Stemmer


class TextPreprocessor:
    def __init__(self, language='english', use_lemmatization=False, use_stemming=True):
        """Initialize the TextPreprocessor.

        Args:
            language (str, optional): The language of the text data. Defaults to 'english'.
            use_lemmatization (bool, optional): Whether to perform lemmatization. Defaults to False.
            use_stemming (bool, optional): Whether to perform stemming. Defaults to True.
        """
        self.language = language
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        if language == 'ukrainian':
            self.__setup_ukrainian_language()
        else:
            self.__setup_default_language()

    @staticmethod
    def __tokenize_text(text):
        words = nltk.tokenize.word_tokenize(text.lower())
        return list(filter(lambda word: word not in string.punctuation, words))

    def __setup_ukrainian_language(self):
        stopwords_ua = pandas.read_csv("../../resources/stopwords_ua.txt", header=None, names=['stopwords'])
        self.stop_words = list(stopwords_ua.stopwords)
        if self.use_stemming:
            self.stemmer = uk_stemmer.UkStemmer()
            self.stem = self.stemmer.stem_word
        if self.use_lemmatization:
            self.lemmatizer = pymorphy2.MorphAnalyzer(lang='uk')
            self.lemmatize = lambda word: self.lemmatizer.parse(word)[0].normal_form

    def __setup_default_language(self):
        self.stop_words = set(nltk.corpus.stopwords.words(self.language))
        if self.use_stemming:
            self.stemmer = nltk.stem.PorterStemmer()
            self.stem = self.stemmer.stem
        if self.use_lemmatization:
            self.lemmatizer = nltk.stem.WordNetLemmatizer()
            self.lemmatize = self.lemmatizer.lemmatize
            self.stemmer = nltk.stem.PorterStemmer()

    def __remove_stopwords(self, words):
        return list(filter(lambda word: word not in self.stop_words, words))

    def __stem_words(self, words):
        return map(self.stem, words)

    def __lemmatize_words(self, words):
        return map(self.lemmatize, words)

    def preprocess(self, text):
        """Preprocess the input text.

        Tokenizes the text, removes stopwords, and optionally performs lemmatization and/or stemming.

        Args:
            text (str): The input text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        tokens = self.__tokenize_text(text)
        tokens = self.__remove_stopwords(tokens)
        if self.use_lemmatization:
            tokens = self.__lemmatize_words(tokens)
        if self.use_stemming:
            tokens = self.__stem_words(tokens)
        return ' '.join(tokens)
