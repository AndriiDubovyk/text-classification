from text_classifier_ml import TextClassifier

# Instantiate the TextClassifier
classifier = TextClassifier()

# Train the classifier with some example data
categories = ["sports", "politics", "technology"]
texts = [
    "The football match was exciting and intense.",
    "The government passed a new law on taxation.",
    "The latest smartphone has impressive features."
]
classifier.train(categories, texts)

# Classify some new texts
new_texts = [
    "A new bill regarding healthcare was proposed.",
    "A soccer tournament is happening next week.",
    "A breakthrough in artificial intelligence research was announced."
]
predicted_categories = classifier.classify(new_texts)

# Print the predicted categories for the new texts
for text, category in zip(new_texts, predicted_categories):
    print(f"Text: '{text}' -> Predicted Category: '{category}'")
