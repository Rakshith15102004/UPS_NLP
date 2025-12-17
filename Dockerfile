# 1. Start with a lightweight Python base
FROM python:3.9-slim

# 2. Install FFmpeg (REQUIRED for Whisper audio processing)
# We update the package list, install ffmpeg, and clean up to keep the file small
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 3. Create a folder inside the container
WORKDIR /app

# 4. Copy the requirements file first
# This helps Docker cache the installation so it's faster next time
COPY requirements.txt .

# 5. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6. Pre-download NLTK data
# This prevents the app from crashing trying to download data during the first request
RUN python -m nltk.downloader punkt stopwords

# 7. Copy all your code and JSON files into the container
COPY . .

# 8. Open the port for the API
EXPOSE 8000

# 9. Run the application
# We use a 300-second timeout because Whisper models take time to load
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]