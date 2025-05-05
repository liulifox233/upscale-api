# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# Copy the project files
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY weights/ ./weights/

# Install Python dependencies
RUN uv sync

# Make port available for the app if needed
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0"]