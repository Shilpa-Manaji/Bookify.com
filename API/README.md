# Sentiment Analysis REST API

This project provides a Flask REST API to serve a sentiment analysis model with TF-IDF preprocessing.

## Project Structure

- `main.py`: Flask app entry point with `/predict` endpoint.
- `model_loader.py`: Loads the saved model and TF-IDF vectorizer.
- `preprocess.py`: Preprocessing functions for input text.
- `requirements.txt`: Python dependencies.

## Setup & Run

1. Create and activate a virtual environment:

```bash
python -m venv obs
obs\Scripts\activate


![img.png](img.png)