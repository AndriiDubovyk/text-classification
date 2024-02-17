import string

import nltk
import pandas
import uk_stemmer  # pip install git+https://github.com/Desklop/Uk_Stemmer


def tokenize_text(text):
    words = nltk.tokenize.word_tokenize(text.lower())
    return list(filter(lambda word: word not in string.punctuation, words))


def remove_stopwords(words, language='english'):
    if language == 'ukrainian':
        stopwords_ua = pandas.read_csv("stopwords_ua.txt", header=None, names=['stopwords'])
        stop_words = list(stopwords_ua.stopwords)
    else:
        stop_words = set(nltk.corpus.stopwords.words(language))
    return list(filter(lambda word: word not in stop_words, words))


def stem_words(words, language='english'):
    if language == 'ukrainian':
        stemmer = uk_stemmer.UkStemmer()
        stemmed_words = map(stemmer.stem_word, words)
    else:
        stemmer = nltk.stem.PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words


def preprocess_text(text, language='english'):
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens, language)
    tokens = stem_words(tokens, language)
    return ' '.join(tokens)


# Example
sample = "This is a sample text for preprocessing. It involves tokenization, removing stopwords, and lemmatization."
sample_ua = "Це зразок тексту для попередньої обробки. Він включає токенізацію, видалення стоп-слів та лематизацію."
print(preprocess_text(sample))
print(preprocess_text(sample_ua, language='ukrainian'))
