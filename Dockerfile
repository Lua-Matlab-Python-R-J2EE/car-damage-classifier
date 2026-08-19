# Lightweight Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Copy application files
COPY README.md .
COPY config.py .
COPY core/ ./core/
COPY utils/ ./utils/
COPY src/ ./src/
COPY streamlit_app/ ./streamlit_app/

# Expose Streamlit's default port
EXPOSE 8501

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Start the application
CMD ["streamlit", "run", "streamlit_app/app.py"]
