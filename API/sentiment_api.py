from flask import Flask, request, jsonify
from model_loader import load_sentiment_model_and_vectorizer
from preprocess import preprocess_text

app = Flask(__name__)

model, vectorizer = load_sentiment_model_and_vectorizer()

@app.route('/predict-sentiment', methods=['POST'])
def predict_sentiment():
    data = request.get_json(force=True)
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'Text is required'}), 400

    sequence = preprocess_text(text, vectorizer)
    prediction = model.predict(sequence)
    score = float(prediction[0][0])
    label = 'positive' if score >= 0.5 else 'negative'
    return jsonify({'label': label, 'score': score})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
