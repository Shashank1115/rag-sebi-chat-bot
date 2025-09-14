FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# ---- NOTE: no apt-get to avoid long downloads ----

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
EXPOSE 5001
CMD ["python", "backend.py"]
