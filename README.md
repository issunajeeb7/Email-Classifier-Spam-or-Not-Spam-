# Email Spam Detector

A machine learning-based email spam detection system built with Python, Streamlit, and scikit-learn. The application uses Natural Language Processing (NLP) and a Multinomial Naive Bayes classifier to identify spam emails with high accuracy.

## Features

- 🎯 Real-time spam detection for email messages
- 📊 Confidence score for each prediction
- 💻 Modern, minimalist web interface
- 🔍 Advanced text preprocessing with email-specific feature extraction
- 📈 High accuracy using TF-IDF and Naive Bayes classification

## Tech Stack

- **Python** - Core programming language
- **Streamlit** - Web interface framework
- **scikit-learn** - Machine learning library
- **NLTK** - Natural Language Processing toolkit
- **pandas** - Data manipulation and analysis
- **joblib** - Model serialization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/issunajeeb7/Email-Classifier-Spam-or-Not-Spam-
cd email-spam-detector
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model (if not already trained):
```bash
python nb.py
```

2. Run the web application:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the displayed URL (typically http://localhost:8501)

4. Enter your email message in the text area and get instant spam detection results

## Model Details

The spam detection model utilizes:
- TF-IDF Vectorization with n-grams (1-3)
- Multinomial Naive Bayes classifier
- Custom preprocessing including:
  - Email-specific feature extraction (URLs, currency symbols, numbers)
  - Spam-related keyword analysis
  - Text normalization and cleaning
  - Stop word removal (with exceptions for important spam-related terms)

## Project Structure

```
├── app.py           # Streamlit web application
├── nb.py            # Model training and preprocessing
├── spam_model.joblib # Trained model
├── spam.csv         # Dataset
└── README.md        # Project documentation
```

## Performance

The model is evaluated using:
- 5-fold cross-validation
- Train/test split (80/20)
- Classification metrics including accuracy, precision, recall, and F1-score

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
