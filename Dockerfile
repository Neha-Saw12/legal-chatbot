# Use Python 3.11 slim image as base
# This provides a smaller image size compared to full Python images
FROM python:3.11-slim

# Set working directory inside container
# All subsequent commands will be run from this directory
WORKDIR /app

# Set environment variables for Python
# PYTHONUNBUFFERED ensures Python output is sent straight to terminal without buffering
# PYTHONDONTWRITEBYTECODE prevents Python from writing pyc files to disk
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for some Python packages
# gcc is needed for compiling some packages
# musl-dev provides C library headers
RUN apt-get update && apt-get install -y \
    gcc \
    musl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
# This way, if only code changes, dependencies won't be reinstalled
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size by not storing package cache
# Using --upgrade to ensure latest versions of packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
# Copy everything except what's in .dockerignore
COPY . .

# Create non-root user for security
# Running as non-root user is a security best practice
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port for the application
# FastAPI will run on port 8000 by default
EXPOSE 8000

# Define the command to run the application
# Using uvicorn as the ASGI server with production settings
# --host 0.0.0.0 makes the app accessible from outside the container
# --port 8000 specifies the port
# --workers 4 uses multiple worker processes for better performance
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]