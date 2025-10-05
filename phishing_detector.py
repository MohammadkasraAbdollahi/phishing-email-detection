import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from bs4 import BeautifulSoup
from scipy.sparse import hstack
import matplotlib.pyplot as plt


def clean_html(text):
    """Remove HTML tags from email body."""
    return BeautifulSoup(text, "html.parser").get_text()


def preprocess_data(df):
    # Fill missing values
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('').apply(clean_html)

    # Combine subject and body
    df['text'] = df['subject'] + ' ' + df['body']

    return df


def extract_features(df, vectorizer=None):
    # TF-IDF Vectorization
    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        X_text = vectorizer.fit_transform(df['text'])
    else:
        X_text = vectorizer.transform(df['text'])

    # Add number of URLs as a numeric feature
    urls = np.array(df['urls']).reshape(-1, 1)
    X = hstack([X_text, urls])

    return X, vectorizer


def train_and_evaluate(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Train model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nROC AUC Score:", roc_auc_score(y_test, y_prob))

    # Plot confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.show()

    return model


def show_top_words(model, vectorizer, top_n=10):
    """Display top indicative words for phishing and legitimate classes."""
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0][:len(feature_names)]

    top_pos = np.argsort(coefs)[-top_n:]
    top_neg = np.argsort(coefs)[:top_n]

    print("\nTop phishing words:")
    for i in reversed(top_pos):
        print(f"  {feature_names[i]}: {coefs[i]:.4f}")

    print("\nTop legitimate words:")
    for i in top_neg:
        print(f"  {feature_names[i]}: {coefs[i]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Phishing Email Detector")
    parser.add_argument('--data', type=str, required=True,
                        help="Path to CSV dataset (with 'subject', 'body', 'urls', 'label' columns)")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)

    # Log class distribution
    print("Label distribution:\n", df['label'].value_counts())

    # Preprocess
    df = preprocess_data(df)

    # Extract features
    X, vectorizer = extract_features(df)
    y = df['label']

    # Train and evaluate
    model = train_and_evaluate(X, y)

    # Show top words
    show_top_words(model, vectorizer)


if __name__ == '__main__':
    main()
