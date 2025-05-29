import os
from tensorflow.keras.models import load_model
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'Models'))

def load_model_and_tokenizer():
    model_path = os.path.join(MODEL_DIR, 'final_lstm_model.h5')
    tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')

    model = load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    return model, tokenizer
