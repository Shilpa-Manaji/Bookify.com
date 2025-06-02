# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirement and app code
COPY requirements.txt .
COPY recommendation_api.py .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code and Models folder
COPY . /app

# Set environment variables
ENV MODEL_PATH=RecommendationModels/content_based_model.pkl
ENV LOG_LEVEL=info

# Expose Flask port
EXPOSE 8000

# Start the Flask app
CMD ["python", "recommendation_api.py"]