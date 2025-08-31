# # Use an official Python runtime as a parent image
# FROM python:3.10-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the requirements file into the container at /app
# COPY requirements.txt .

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application code into the container at /app
# COPY . .

# # Make port 5001 available to the world outside this container
# EXPOSE 5001

# # Define environment variable
# ENV NAME World

# # Run backend.py when the container launches
# CMD ["python", "backend.py"] 
# #
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
# This includes your 'data' folder which is needed for ingestion
COPY . .

# --- THIS IS THE CRITICAL FIX ---
# Run the data ingestion script to build the main vector store
# This runs only once when the Docker image is built on the server
RUN python ingest.py

# Expose the port that the Flask app runs on
EXPOSE 5001

# Define the command to run your application's web server
CMD ["python", "backend.py"]
