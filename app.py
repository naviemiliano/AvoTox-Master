from flask import Flask, request, jsonify, render_template
import joblib
import re
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and vectorizer
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('multi_output_model.pkl')

app = Flask(__name__)

# Preprocessing function (same as in your training process)
def preprocess_text(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s.,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Route to serve the index.html page
@app.route('/')
def index():
    return render_template('index.html')  # Flask will look for index.html in the templates folder

# Define the API route to check toxicity
@app.route('/check_toxicity', methods=['POST'])
def check_toxicity():
    data = request.get_json()  # Receive the input from the front end
    comment = data.get('comment')

    # Preprocess the comment
    preprocessed_comment = preprocess_text(comment)

    # Vectorize the comment
    comment_vector = vectorizer.transform([preprocessed_comment])

    # Predict the toxicity levels using the trained model
    prediction = model.predict(comment_vector)

    # Create a response dictionary with the results (convert numpy.int64 to int)
    response = {
        'toxicity': int(prediction[0][0]),
        'severe_toxic': int(prediction[0][1]),
        'obscene': int(prediction[0][2]),
        'threat': int(prediction[0][3]),
        'insult': int(prediction[0][4]),
        'identity_hate': int(prediction[0][5])
    }

    return jsonify(response)  # Send the results back to the front end

if __name__ == '__main__':
    app.run(debug=True)
