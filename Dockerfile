# DenLsNet Deployment Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_deployment.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_deployment.txt

# Copy application files
COPY app_deployment.py .
COPY model/ ./model/
COPY config/ ./config/
COPY .streamlit/ ./.streamlit/

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "app_deployment.py", "--server.port=8501", "--server.address=0.0.0.0"]
