# 🛡️ Phishing Email Detection Using Machine Learning

This project builds a machine learning pipeline to **detect phishing emails** based on their content and structure. It uses basic natural language processing (NLP) and metadata features (e.g., number of URLs) to classify emails as **phishing** or **legitimate**.

---

## 📦 Features

- Combines email **subject** and **body** into a single text input.
- Cleans and strips **HTML tags** from email body.
- Converts text to vectors using **TF-IDF**.
- Adds **number of URLs** as a numerical feature.
- Uses **Logistic Regression** for classification.
- Handles **class imbalance** via `class_weight='balanced'`.
- Outputs detailed evaluation:
  - Classification report
  - Confusion matrix
  - ROC AUC score
- Shows top words that indicate phishing or legitimate emails.

---

## 📁 Dataset Format

Your CSV file must contain the following columns:

| Column  | Description                             |
|---------|-----------------------------------------|
| subject | Email subject (text)                    |
| body    | Email body (HTML or plain text)         |
| urls    | Number of URLs in the email (integer)   |
| label   | Target label: `1` = phishing, `0` = legit|

Example row:

```csv
subject,body,urls,label
"Win a free iPhone!","<html>Click <a href='http://scam.com'>here</a> to claim now!</html>",1,1
🛠️ Installation
bash
Copy code
git clone https://github.com/yourusername/phishing-email-detection.git
cd phishing-email-detection

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install required packages
pip install -r requirements.txt
🚀 Usage
bash
Copy code
python phishing_detector.py --data data/emails.csv
You’ll see output like:

Label distribution

Classification metrics

Confusion matrix (also visualized)

Top phishing/legitimate keywords

##🧪 Example Output
text
Copy code
Label distribution:
 0    500
 1    150

Classification Report:
              precision    recall  f1-score   support
         0       0.98      0.99      0.98       100
         1       0.99      0.97      0.98        30

Confusion Matrix:
[[99  1]
 [ 1 29]]

ROC AUC Score: 0.9942

Top phishing words:
  replica: 1.382
  watches: 1.142
  click: 1.101
  http: 1.076
  win: 0.963

Top legitimate words:
  thanks: -0.921
  wrote: -0.865
  update: -0.755
  python: -0.672
##🔍 Requirements
You’ll find all dependencies in requirements.txt. Key packages:

scikit-learn

pandas

numpy

scipy

beautifulsoup4

matplotlib

##🧠 Ideas for Improvement
Add support for email attachments or headers

Train a neural network or fine-tune a transformer (e.g., DistilBERT)

Build a Streamlit or FastAPI app for real-time email classification

Add explainability using SHAP or LIME

Use additional features like:

Domain reputation

IP addresses

Language detection

Email length & read-time

##📄 License
MIT License © 2025 [Your Name]

##🤝 Contributions
PRs welcome! If you'd like to suggest new features or fixes, feel free to open an issue or submit a pull request.

