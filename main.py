from classifier import TextClassifier

categories = ["Technology", "Sports", "Politics"]

texts = [
    "Artificial intelligence is revolutionizing the tech industry. Companies are investing heavily in AI research and development.",
    "The Super Bowl is one of the most-watched sporting events in the United States. Football fans eagerly anticipate this annual championship game.",
    "The latest election results have sparked debates across the nation. Political analysts are dissecting the outcomes and their implications.",
    "Machine learning algorithms are being used to predict customer behavior in e-commerce. This helps businesses personalize user experiences and increase sales.",
    "Basketball is a popular sport worldwide. NBA players are admired for their skills and athleticism on the court.",
    "Virtual reality technology is changing the way we interact with computers. VR headsets offer immersive experiences for gaming and simulations."
]
classifier = TextClassifier(language='english', use_lemmatization=False, use_stemming=True)
result = classifier.classify(categories, texts)
print("Result:", result)
