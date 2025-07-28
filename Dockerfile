# Use an official Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY ./app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app .

# Create necessary directories
RUN mkdir -p uploads templates

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose the Flask port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/ || exit 1

# Run the Flask application
CMD ["python", "index.py"]