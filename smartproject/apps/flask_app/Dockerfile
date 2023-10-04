FROM python:3.9.13

# Set working directory
WORKDIR /app

# Copy requirements.txt from the smartproject/apps/flask_app/ directory
COPY apps/flask_app/requirements.txt .

COPY .env   ./

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY apps/flask_app/ .

# Expose the Flask app on port 5000
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
