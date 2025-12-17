# 1. Use Python 3.9 Slim (Standard for data science apps)
FROM python:3.9-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install system dependencies (REQUIRED for Audio processing)
# ffmpeg: converts audio formats
# libsndfile1: allows python to read audio files safely
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements file first (for caching)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Pre-download NLTK data (prevents crashes on startup)
RUN python -m nltk.downloader punkt stopwords

# 7. Copy all your application files
COPY . .

# 8. Expose the port
EXPOSE 8000

# 9. Run the application
# We use a 300-second timeout to give the model time to load if needed
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]
