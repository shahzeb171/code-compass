# Use lightweight Python base image
FROM python:3.11-slim

# Prevents Python from writing pyc files to disk & keeps output unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies if needed (faiss, build tools, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose app port (e.g., Gradio usually on 7860, FastAPI on 8000)
EXPOSE 7860

# Start your app (adjust command for your framework)
CMD ["python", "main.py"]
