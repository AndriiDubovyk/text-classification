import math

from preprocessor import TextPreprocessor


class TextClassifier:
    def __init__(self, language='english', use_lemmatization=False, use_stemming=True):
        self.preprocessor = TextPreprocessor(language, use_lemmatization, use_stemming)

    @staticmethod
    def __calculate_tf(text):
        words = text.split()
        word_count = {}
        total_words = len(words)
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        tf = {word: count / total_words for word, count in word_count.items()}
        return tf

    @staticmethod
    def __calculate_idf(texts):
        word_document_count = {}
        total_documents = len(texts)
        for text in texts:
            words = set(text.split())
            for word in words:
                word_document_count[word] = word_document_count.get(word, 0) + 1
        idf = {word: math.log10(total_documents / (count + 1)) for word, count in word_document_count.items()}
        return idf

    @staticmethod
    def __calculate_tfidf(texts):
        tfidf = []
        idf = TextClassifier.__calculate_idf(texts)

        for text in texts:
            tf = TextClassifier.__calculate_tf(text)
            tfidf_doc = {word: tf[word] * idf[word] for word in tf}
            tfidf.append(tfidf_doc)

        return tfidf

    @staticmethod
    def __cosine_similarity(vector1, vector2):
        dot_product = sum(vector1.get(word, 0) * vector2.get(word, 0) for word in set(vector1) & set(vector2))
        magnitude1 = math.sqrt(sum(value ** 2 for value in vector1.values()))
        magnitude2 = math.sqrt(sum(value ** 2 for value in vector2.values()))
        return dot_product / (magnitude1 * magnitude2)

    def __calculate_similarities(self, query, texts):
        processed_query = self.preprocessor.preprocess(query)
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        tfidf = self.__calculate_tfidf(processed_texts)
        query_tfidf = self.__calculate_tf(processed_query)
        similarities = [self.__cosine_similarity(query_tfidf, doc_tfidf) for doc_tfidf in tfidf]
        return similarities

    def classify(self, categories, texts):
        category_to_text_similarities = []
        for category in categories:
            similarities = self.__calculate_similarities(category, texts)
            category_to_text_similarities.append(similarities)
        text_to_category_similarities = list(zip(*category_to_text_similarities))
        best_categories_indices = [value.index(max(value)) for value in text_to_category_similarities]
        return [categories[index] for index in best_categories_indices]
