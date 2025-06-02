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
```
**Sentiment.py result**

![img.png](img.png)

**Recommendation.py output**
```bash
curl -X POST http://localhost:8001/recommend -H "Content-Type: application/json" -d "{\"user_id\": \"123\"}"
```
![img_1.png](img_1.png)

**Recommendation_api docker image**
```bash
(obs) PS C:\Users\HP\Desktop\Bookify\API> docker build -t bookify-recommend .
```
```bash
(obs) PS C:\Users\HP\Desktop\Bookify\API> docker run -p 8001:8000 bookify-recommend
```
![img_3.png](img_3.png)

Docker results
```bash
curl -X POST http://localhost:8001/recommend -H "Content-Type: application/json" -d "{\"user_id\": \"123\"}"
```
![img_4.png](img_4.png)

**Sentiment_api docker image**
```bash
(obs) PS C:\Users\HP\Desktop\Bookify> docker build -t bookify-sentiment -f "API/Dockerfile(sentiment)" .
```
```bash
docker run -p 8002:8000 bookify-sentiment
```
![img_5.png](img_5.png)!

Docker results for sentiment
```bash
curl -X POST http://localhost:8002/predict-sentiment -H "Content-Type: application/json" -d "{\"text\":\"I love this product!\"}"
```
![img_6.png](img_6.png)

Result for health
```bash
curl http://localhost:8002/health
```
![img_7.png](img_7.png)