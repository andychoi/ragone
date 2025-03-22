# Use official Python image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (✅ Fix SSL issue)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && update-ca-certificates

# Copy the requirements file
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory to /app
COPY app /app

# Copy the .env file (Ensure it's included in .dockerignore)
COPY .env /app/.env

# Expose API port
EXPOSE 8000

# Run the FastAPI service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]