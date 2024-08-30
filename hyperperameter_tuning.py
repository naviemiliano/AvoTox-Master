import pandas as pd
import contractions
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.multioutput import MultiOutputClassifier
import joblib

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
  text = text.lower()
  # Expand contractions
  text = contractions.fix(text)
  # Remove HTML tags
  text = re.sub(r'<.*?>', '', text)
  # Remove URLs
  text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
  # Remove email addresses
  text = re.sub(r'\S+@\S+', '', text)
  # Remove hashtags
  text = re.sub(r'#\w+', '', text)
  # Remove mentions
  text = re.sub(r'@\w+', '', text)
  # Remove special characters except punctuation
  text = re.sub(r'[^a-zA-Z\s.,]', '', text)
  # Remove extra whitespace
  text = re.sub(r'\s+', ' ', text).strip()
  # Tokenize text
  words = text.split()
  # Remove stop words
  stop_words = set(stopwords.words('english'))
  words = [word for word in words if word not in stop_words]
  # Lemmatize words
  lemmatizer = WordNetLemmatizer()
  words = [lemmatizer.lemmatize(word) for word in words]
  # Join words to a single string
  text = ' '.join(words)
  # print(text)
  return text


# Load the files into pandas DataFrames
train_data = pd.read_csv('csv-files/train.csv')
test_data = pd.read_csv('csv-files/test.csv')
test_labels = pd.read_csv('csv-files/test_labels.csv')

# Dislay the first few rows of the datasets
print("train_data")
print(train_data.head())
print("test_data")
print(test_data.head())
print("test_labels")
print(test_labels.head())


def train_custom_model(rf_n_estimators=100, rf_max_depth=10, xgb_learning_rate=0.1, xgb_max_depth=6):
    vectorizer = TfidfVectorizer(max_features=10000)

    # Apply preprocessing (Assuming preprocess_text function and vectorizer are defined)
    train_data['comment_text'] = train_data['comment_text'].apply(preprocess_text)
    test_data['comment_text'] = test_data['comment_text'].apply(preprocess_text)
    
    # Vectorize the text data
    X_train = vectorizer.fit_transform(train_data['comment_text'])
    y_train = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Initialize models with user-specified hyperparameters
    rf_model = RandomForestClassifier(random_state=42, n_estimators=rf_n_estimators, max_depth=rf_max_depth, class_weight='balanced')
    xgb_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, learning_rate=xgb_learning_rate, max_depth=xgb_max_depth)
    
    # Combine the models in a VotingClassifier
    ensemble_model = VotingClassifier(estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model)
    ], voting='soft')
    
    multi_output_model = MultiOutputClassifier(ensemble_model)
    
    # Fit the ensemble model
    multi_output_model.fit(X_train, y_train)
    
    # Evaluate the model on the validation set
    y_val_pred = multi_output_model.predict(X_val)
    print('Validation Set Results')
    print(classification_report(y_val, y_val_pred, target_names=y_train.columns, zero_division=0))


def user_interface():
    print("Welcome to the Toxic Comment Classifier Hyperparameter Tuning")
    print("Please select the hyperparameters for the model:")
    
    # Get user input for RandomForest hyperparameters
    rf_n_estimators = int(input("Enter number of trees for RandomForest (n_estimators) [Default: 100]: ") or 100)
    rf_max_depth = input("Enter maximum depth of the tree for RandomForest (max_depth) [Default: 10]: ")
    rf_max_depth = int(rf_max_depth) if rf_max_depth else 10
    
    # Get user input for XGBoost hyperparameters
    xgb_learning_rate = float(input("Enter learning rate for XGBoost (learning_rate) [Default: 0.1]: ") or 0.1)
    xgb_max_depth = int(input("Enter maximum depth of the tree for XGBoost (max_depth) [Default: 6]: ") or 6)
    
    # Train the model with selected hyperparameters
    train_custom_model(rf_n_estimators, rf_max_depth, xgb_learning_rate, xgb_max_depth)
    
if __name__ == "__main__":
    user_interface()
