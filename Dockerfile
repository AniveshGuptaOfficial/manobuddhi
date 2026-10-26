# Image for the RQ worker (and, from F8, the API). Source code is mounted at /app in dev.
FROM python:3.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
