#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Function to load and preprocess the data
def load_data(file_path):
    # Read CSV file
    df = pd.read_csv(file_path, encoding="ISO-8859-1")

    # Splitting into training and testing sets
    train = df[df['Date'] < '20150101']
    test = df[df['Date'] > '20141231']

    # Preprocessing headlines
    train_headlines = preprocess_headlines(train)
    test_headlines = preprocess_headlines(test)

    return train_headlines, test_headlines, train['Label'], test['Label']

# Function to preprocess headlines
def preprocess_headlines(df):
    # Combine headlines into one string
    headlines = df.apply(lambda row: ' '.join(str(x).lower() for x in row[2:27]), axis=1)
    return headlines

# Function to train the model
def train_model(train_headlines, labels):
    # Vectorize the headlines
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    train_dataset = vectorizer.fit_transform(train_headlines)

    # Train RandomForest Classifier
    random_forest_classifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
    random_forest_classifier.fit(train_dataset, labels)

    return random_forest_classifier, vectorizer

# Function to test the model
def test_model(model, vectorizer, test_headlines, labels):
    # Transform test dataset
    test_dataset = vectorizer.transform(test_headlines)

    # Make predictions
    predictions = model.predict(test_dataset)

    # Evaluate the model
    evaluate_model(labels, predictions)

# Function to evaluate the model
def evaluate_model(true_labels, predicted_labels):
    # Confusion matrix
    matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(matrix)

    # Accuracy score
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("\nAccuracy Score:", accuracy)

    # Classification report
    report = classification_report(true_labels, predicted_labels)
    print("\nClassification Report:")
    print(report)

# Main function
def main():
    # Load and preprocess data
    train_headlines, test_headlines, train_labels, test_labels = load_data('Data.csv')

    # Train the model
    model, vectorizer = train_model(train_headlines, train_labels)

    # Test the model
    test_model(model, vectorizer, test_headlines, test_labels)

if __name__ == "__main__":
    main()
