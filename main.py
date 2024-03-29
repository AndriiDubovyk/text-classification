from sklearn.datasets import fetch_20newsgroups

from src.models.text_classifier_ml_tf import TextClassifierTF

if __name__ == "__main__":
    newsgroups = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    X_train, Y_train = newsgroups.data, newsgroups.target

    # Initialize the classifier
    classifier_tf = TextClassifierTF()

    # # Train the classifier
    # classifier_tf.train(X_train, Y_train, newsgroups.target_names)
    #
    # # Save the trained model
    # classifier_tf.save_model()
    #
    # Load the model
    classifier_tf.load_model()

    # Predict category for user input
    user_text = input("Enter some text for prediction: ")
    predicted_category_index = classifier_tf.predict_category(user_text)
    print("Predicted category index:", predicted_category_index)
