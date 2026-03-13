FROM python:3.11-slim

# Install system dependencies including libwebp
RUN apt-get update && apt-get install -y \
    libwebp7 \
    libwebpdemux2 \
    libwebpmux3 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Command to run the application
CMD gunicorn server:app --timeout 120 --workers 1 --threads 2
