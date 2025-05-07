from flask import Flask, request, jsonify, abort
import pandas as pd
import joblib
import os
from functools import wraps
import jwt
import time
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Setup for rate limiting
limiter = Limiter(get_remote_address, app=None)

# Flask app initialization
app = Flask(__name__)

# Secret key for JWT
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key')

# Relative Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'Models')
DATA_DIR = os.path.join(BASE_DIR, '..', 'Database', 'Cleaned_Datasets')

USER_USER_MODEL_PATH = os.path.join(MODEL_DIR, 'user_user_cf_knn_model.pkl')
ITEM_ITEM_MODEL_PATH = os.path.join(MODEL_DIR, 'item_item_cf_knn_model.pkl')
CONTENT_BASED_MODEL_PATH = os.path.join(MODEL_DIR, 'content_based_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

BOOKS_DATA_PATH = os.path.join(DATA_DIR, 'books_cleaned.csv')

# Load Models into Variables
user_user_model = joblib.load(USER_USER_MODEL_PATH)
item_item_model = joblib.load(ITEM_ITEM_MODEL_PATH)
content_based_model = joblib.load(CONTENT_BASED_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Load Dataset into Variable
books_df = pd.read_csv(BOOKS_DATA_PATH)
book_title_to_index = pd.Series(books_df.index, index=books_df['title'].str.lower()).to_dict()

# Helper function to verify JWT token
def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 403
        try:
            data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
            current_user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 403
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 403
        return f(current_user_id, *args, **kwargs)
    return decorator

# ROUTE: User-User Recommendation
@app.route('/api/recommendations/<int:user_id>', methods=['GET'])
@limiter.limit("100 per hour")  # Rate limiting
@token_required
def recommend_user_user(current_user_id, user_id):
    if current_user_id != user_id:
        return jsonify({'error': 'Unauthorized access'}), 403

    top_n = int(request.args.get('top_n', 5))

    try:
        distances, indices = user_user_model.kneighbors(user_user_model.X[user_id], n_neighbors=top_n + 1)
        recommendations = []
        for i in indices.flatten():
            if i != user_id:
                book_ids = user_user_model.X[i].nonzero()[1]
                recommended_titles = books_df.iloc[book_ids]['title'].tolist()
                recommendations.extend(recommended_titles)
        return jsonify({'user_id': user_id, 'recommendations': recommendations[:top_n]})
    except Exception as e:
        return jsonify({'error': str(e)})

# ROUTE: Top K Recommendations for User
@app.route('/api/recommendations/<int:user_id>/top_k', methods=['GET'])
@limiter.limit("100 per hour")  # Rate limiting
@token_required
def recommend_top_k(current_user_id, user_id):
    if current_user_id != user_id:
        return jsonify({'error': 'Unauthorized access'}), 403

    top_k = int(request.args.get('count', 10))

    try:
        distances, indices = user_user_model.kneighbors(user_user_model.X[user_id], n_neighbors=top_k + 1)
        recommendations = []
        for i in indices.flatten():
            if i != user_id:
                book_ids = user_user_model.X[i].nonzero()[1]
                recommended_titles = books_df.iloc[book_ids]['title'].tolist()
                recommendations.extend(recommended_titles)
        return jsonify({'user_id': user_id, 'top_k_recommendations': recommendations[:top_k]})
    except Exception as e:
        return jsonify({'error': str(e)})

# ROUTE: Item-Item Recommendation
@app.route('/api/recommendations/item-item', methods=['GET'])
@limiter.limit("100 per hour")  # Rate limiting
@token_required
def recommend_item_item(current_user_id):
    book_title = request.args.get('title', '').lower()
    top_n = int(request.args.get('top_n', 5))

    try:
        idx = book_title_to_index.get(book_title)
        if idx is None:
            return jsonify({'error': 'Book title not found'}), 404

        distances, indices = item_item_model.kneighbors(item_item_model.X[idx], n_neighbors=top_n + 1)
        recommended_books = books_df.iloc[indices.flatten()]['title'].tolist()
        return jsonify({'input_title': book_title, 'recommendations': recommended_books[1:]})
    except Exception as e:
        return jsonify({'error': str(e)})

# ROUTE: Content-Based Recommendation
@app.route('/api/recommendations/content-based', methods=['GET'])
@limiter.limit("100 per hour")  # Rate limiting
@token_required
def recommend_content_based(current_user_id):
    book_title = request.args.get('title', '').lower()
    top_n = int(request.args.get('top_n', 5))

    try:
        idx = book_title_to_index.get(book_title)
        if idx is None:
            return jsonify({'error': 'Book title not found'}), 404

        book_vector = content_based_model[idx]
        cosine_similarities = book_vector.dot(content_based_model.T).toarray().flatten()
        similar_indices = cosine_similarities.argsort()[::-1][1:top_n + 1]
        recommended_titles = books_df.iloc[similar_indices]['title'].tolist()

        return jsonify({'input_title': book_title, 'recommendations': recommended_titles})
    except Exception as e:
        return jsonify({'error': str(e)})

# ROUTE: Get Book Details
@app.route('/api/book/<int:book_id>', methods=['GET'])
@limiter.limit("100 per hour")  # Rate limiting
@token_required
def get_book_details(current_user_id, book_id):
    try:
        book = books_df.iloc[book_id]
        book_details = {
            'title': book['title'],
            'author': book['author'],
            'rating': book['rating'],
            'genres': book['genres']
        }
        return jsonify({'book_details': book_details})
    except Exception as e:
        return jsonify({'error': 'Book not found'}), 404

# ROUTE: User Login (for JWT Authentication)
@app.route('/api/auth/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    # Authentication logic (you can replace with your actual logic)
    if username == "user" and password == "password":
        token = jwt.encode({'user_id': 1, 'exp': time.time() + 3600}, app.config['JWT_SECRET_KEY'], algorithm="HS256")
        return jsonify({'token': token})
    return jsonify({'error': 'Invalid credentials'}), 401

# Route for the root endpoint
@app.route('/')
def home():
    return 'Welcome to the Recommendations API!'

# MAIN
if __name__ == '__main__':
    app.run(debug=True)
