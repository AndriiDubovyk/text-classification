from preprocessor import TextPreprocessor

# Example
sample = "This is a sample text for preprocessing. It involves tokenization, removing stopwords, and lemmatization"
sample_ua = "Це зразок тексту для попередньої обробки. Він включає токенізацію, видалення стоп-слів та лематизацію."
default_preprocessor = TextPreprocessor(use_lemmatization=True, use_stemming=False)
default_ukrainian_preprocessor = TextPreprocessor(language='ukrainian', use_lemmatization=True, use_stemming=False)
print(default_preprocessor.preprocess(sample))
print(default_ukrainian_preprocessor.preprocess(sample_ua))
