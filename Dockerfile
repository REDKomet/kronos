FROM python:3.12.9-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /service
RUN git clone https://github.com/shiyu-coder/Kronos /service/kronos && cd /service/kronos && git checkout 67b630e67f6a18c9e9be918d9b4337c960db1e9a
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
ENV PYTHONPATH=/service/kronos PYTHONUNBUFFERED=1
EXPOSE 8000
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]
