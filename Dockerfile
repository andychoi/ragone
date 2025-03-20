# Use official Python image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory to /app
COPY app /app

# Expose API port
EXPOSE 8000

# Run the FastAPI service
CMD ["python", "main.py"]