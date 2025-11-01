Creating a semantic analysis tool like DocuParse-Insights involves several key steps: loading and pre-processing the documents, performing semantic analysis using NLP techniques, applying machine learning models, and finally outputting actionable insights. Below is a simple, yet complete Python program to demonstrate this process. Note that this is a basic example and doesn't include advanced machine learning models due to the complexity, but it gives a foundational overview.

We'll use libraries such as NLTK for NLP and Scikit-learn for a simple machine learning model. Ensure to install necessary packages before running the program:

```bash
pip install nltk scikit-learn
```

Here is the complete Python program:

```python
import os
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def load_documents(directory):
    """Load documents from a directory."""
    documents = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                    documents.append(file.read())
            else:
                print(f"Skipped non-text file: {filename}")
    except Exception as e:
        print(f"Error reading files: {e}")
    return documents

def preprocess_document(document):
    """Preprocess the document text with basic cleaning and tokenization."""
    try:
        # Remove punctuation and numbers
        document = re.sub(r'[^\w\s]', '', document)
        document = re.sub(r'\d+', '', document)

        # Convert to lowercase
        document = document.lower()

        # Tokenize words
        words = word_tokenize(document)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        return ' '.join(words)
    except Exception as e:
        print(f"Error in preprocessing document: {e}")
        return ""

def create_feature_matrix(documents):
    """Create a feature matrix from the list of documents."""
    try:
        vectorizer = CountVectorizer()
        feature_matrix = vectorizer.fit_transform(documents)
        return feature_matrix, vectorizer
    except Exception as e:
        print(f"Error in creating feature matrix: {e}")
        return None, None

def train_model(feature_matrix, labels):
    """Train a simple machine learning model."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.3, random_state=42)
        model = MultinomialNB()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print("Accuracy: ", accuracy_score(y_test, predictions))
        print("\nClassification Report:\n", classification_report(y_test, predictions))
    except Exception as e:
        print(f"Error in training model: {e}")

def main():
    documents = load_documents('documents')

    if not documents:
        print("No documents to process.")
        return

    print("Preprocessing documents...")
    processed_docs = [preprocess_document(doc) for doc in documents]

    print("Creating feature matrix...")
    feature_matrix, vectorizer = create_feature_matrix(processed_docs)

    if feature_matrix is None:
        print("Failed to create feature matrix.")
        return

    # Dummy labels for demonstration; replace with actual labels in a real scenario
    labels = [0, 1, 0]  # assuming binary classification

    print("Training model...")
    train_model(feature_matrix, labels)

if __name__ == "__main__":
    main()
```

### Key Components and Steps

1. **Loading Documents**: The `load_documents` function loads all text documents from a given directory. Non-text files are skipped.

2. **Preprocessing**: The `preprocess_document` function cleans text data by removing punctuation, numbers, and stopwords, and then tokenizes it.

3. **Feature Matrix Creation**: `create_feature_matrix` uses `CountVectorizer` from Scikit-learn to convert text to a numerical feature matrix.

4. **Model Training**: A simple Naive Bayes model is used to demonstrate training a machine learning model on the data. The accuracy and classification report are printed for evaluation.

5. **Error Handling**: Each major function contains try-except blocks to handle possible errors gracefully.

This program offers a skeleton framework to start from, and further enhancements can be made by incorporating more advanced NLP techniques and machine learning algorithms based on specific requirements and the nature of the documents.