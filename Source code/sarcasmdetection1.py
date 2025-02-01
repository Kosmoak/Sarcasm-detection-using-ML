import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pickle

# Function to load and preprocess data
def load_data(file_path):
        data = pd.read_json(file_path, lines=True)
        data["is_sarcastic"] = data["is_sarcastic"].map({0: "Not Sarcasm", 1: "Sarcasm"})
        data = data[["headline", "is_sarcastic"]]
        return data

# Function to train the model
def train_model(data):
    x = np.array(data["headline"])
    y = np.array(data["is_sarcastic"])

    # Vectorize text data
    cv = CountVectorizer()
    X = cv.fit_transform(x)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("Class distribution after SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # Train Bernoulli Naive Bayes model
    model = LogisticRegression()
    model.fit(X_train_resampled, y_train_resampled)
    predictions = model.predict(X_test)

    # Print model accuracy
    print("Model accuracy:", model.score(X_test, y_test))
    print("\n")
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(pd.Series(y_train).value_counts())           # Before SMOTE
    print(pd.Series(y_train_resampled).value_counts()) # After SMOTE


    return model, cv

# Main script
if __name__ == "__main__":
    data = load_data("Sarcasm.json")
    if data is not None:
        model, cv = train_model(data)

        # Save the trained model and vectorizer
        with open("sarcasm_model.pkl", "wb") as f:
            pickle.dump(model, f)

        with open("count_vectorizer.pkl", "wb") as f:
            pickle.dump(cv, f)

        print("\nModel and vectorizer saved successfully!\n")
