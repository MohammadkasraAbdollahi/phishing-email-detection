# Phishing Email Detection Using Machine Learning

## Project Overview  
This project builds a machine learning model to detect phishing emails based on email content and features. It uses a dataset of emails labeled as phishing or legitimate.

## Dataset  
The dataset includes email metadata like sender, receiver, subject, body, number of URLs, and a label indicating phishing (1) or legitimate (0).

## Approach  
- Combined email subject and body into a single text feature  
- Used TF-IDF vectorization to convert text into numerical features  
- Added the number of URLs as a numeric feature  
- Trained a Logistic Regression classifier to detect phishing emails

## Results  
- Achieved 99% accuracy on the test set  
- Precision and recall for phishing class were both 0.99 or higher  
- Top words indicating phishing include "http", "replica", "watches", etc.  
- Top words indicating legitimate emails include "thanks", "wrote", "python", etc.

## How to Run  
1. Clone this repository  
2. Install required packages:  
   ```bash
   pip install pandas scikit-learn numpy scipy
