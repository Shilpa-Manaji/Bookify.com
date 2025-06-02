from flask import Flask, request, jsonify
from model_loader import load_recommendation_model

app = Flask(__name__)

# Load recommendation model
model = load_recommendation_model()

@app.route('/')
def home():
    return "Welcome to the Recommendation API!"

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json(force=True)

    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    try:
        recommendations = model.recommend(user_id)
        return jsonify({"user_id": user_id, "recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)  # listen on 0.0.0.0 and port 8000 inside container
