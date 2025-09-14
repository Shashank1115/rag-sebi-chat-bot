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


# Dockerfile (optimized - minimal system deps)
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

WORKDIR /app

# Minimal system deps - keep this list short. Add only what's required by your wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    gcc \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

EXPOSE 5001

CMD ["python", "backend.py"]
