# Sentiment Analysis REST API

This project provides a Flask REST API to serve a sentiment analysis model with TF-IDF preprocessing.

## Project Structure

- `main.py`: Flask app entry point with `/predict` endpoint.
- `model_loader.py`: Loads the saved model and TF-IDF vectorizer.
- `preprocess.py`: Preprocessing functions for input text.
- `requirements.txt`: Python dependencies.
- `recommendation_api.py`: FastAPI service that handles book recommendations using collaborative filtering or content-based models. 
- `sentiment_api.py`: FastAPI or Flask-based API to analyze review sentiment using a pre-trained model and vectorizer.
- `Dockerfile`: Docker configuration for building the recommendation API image. 
- `Dockerfile(sentiment)`: Docker configuration for building the sentiment analysis API image. 
- `requirements.txt`: Python dependencies.

## Setup & Run

1. Create and activate a virtual environment:

```bash
python -m venv obs
obs\Scripts\activate
```
**Sentiment.py result**

![img.png](outputs%2Fimg.png)!

**Recommendation.py output**
```bash
curl -X POST http://localhost:8001/recommend -H "Content-Type: application/json" -d "{\"user_id\": \"123\"}"
```
![img_1.png](outputs%2Fimg_1.png)

**Recommendation_api docker image**
```bash
(obs) PS C:\Users\HP\Desktop\Bookify\API> docker build -t bookify-recommend .
```
```bash
(obs) PS C:\Users\HP\Desktop\Bookify\API> docker run -p 8001:8000 bookify-recommend
```
![img_2.png](outputs%2Fimg_2.png)

Docker results
```bash
curl -X POST http://localhost:8001/recommend -H "Content-Type: application/json" -d "{\"user_id\": \"123\"}"
```
![img_4.png](outputs%2Fimg_4.png)

**Sentiment_api docker image**
```bash
(obs) PS C:\Users\HP\Desktop\Bookify> docker build -t bookify-sentiment -f "API/Dockerfile(sentiment)" .
```
```bash
docker run -p 8002:8000 bookify-sentiment
```

For Docker registry
```bash
docker pull shilpa2003/bookify-sentiment:latest
docker run -p 8002:8000 shilpa2003/bookify-sentiment:latest
```

![img_5.png](outputs%2Fimg_5.png)

Docker results for sentiment
```bash
curl -X POST http://localhost:8002/predict-sentiment -H "Content-Type: application/json" -d "{\"text\":\"I love this product!\"}"
```
![img_6.png](outputs%2Fimg_6.png)

Result for health
```bash
curl http://localhost:8002/health
```
![img_7.png](outputs%2Fimg_7.png)