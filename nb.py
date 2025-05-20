import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import joblib
import joblib
import os

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
file_path = 'spam.csv'
df = pd.read_csv(file_path, encoding='latin-1')

# Drop unnecessary columns and rename columns
df = df[['email', 'label']]
df.columns = ['message', 'label']

# Convert labels to binary (0 for ham, 1 for spam)
df['label'] = df['label'].astype(int)

# Drop any remaining NaN values
df = df.dropna()

# Verify we have data
print(f"Dataset shape: {df.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Text preprocessing function with enhanced spam features
def preprocess_text(text):
    # Handle non-string values
    if not isinstance(text, str):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Extract email-specific features
    has_url = 1 if 'http' in text or 'www' in text else 0
    has_currency = 1 if any(c in text for c in ['$', '€', '£']) else 0
    has_number = 1 if any(c.isdigit() for c in text) else 0
    
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Expanded spam-related words list
    important_words = {
        'urgent', 'verify', 'free', 'winner', 'win', 'click', 'account',
        'limited', 'prize', 'congratulations', 'claim', 'offer', 'discount',
        'reward', 'guaranteed', 'exclusive', 'riskfree', 'cash', 'bonus',
        'subscribe', 'password', 'deal', 'million', 'lottery', 'security',
        'payment', 'credit', 'loan', 'expire', 'instant', 'risk', 'access',
        'membership', 'confidential', 'important', 'action', 'required'
    }
    
    # Keep important words even if they are stopwords
    words = [word for word in words if word not in stop_words or word in important_words]
    processed_text = ' '.join(words)
    
    # Add email-specific features to the text
    feature_text = f'{processed_text} url_{has_url} currency_{has_currency} number_{has_number}'
    return feature_text

# Apply preprocessing
df['processed_message'] = df['message'].apply(preprocess_text)

# Drop rows with NaN values in labels
df = df.dropna(subset=['label'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_message'], df['label'], test_size=0.2, random_state=42
)

# Create pipeline with TF-IDF and Multinomial Naive Bayes
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 3), min_df=2),
    MultinomialNB(alpha=0.1)
)

# Train the model
model.fit(X_train, y_train)

# Save the trained model
model_path = 'spam_model.joblib'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Average CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})')

# Evaluate on test set
y_pred = model.predict(X_test)
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Function to predict user input
def predict_spam():
    while True:
        user_input = input('\nEnter a message (or "quit" to exit): ')
        if user_input.lower() == 'quit':
            break
        processed_input = preprocess_text(user_input)
        prediction = model.predict([processed_input])[0]
        probability = model.predict_proba([processed_input])[0]
        result = 'Spam' if prediction == 1 else 'Not Spam'
        confidence = probability[1] if prediction == 1 else probability[0]
        print(f'Prediction: {result} (Confidence: {confidence:.2f})')

# Run the prediction interface
if __name__ == '__main__':
    predict_spam()