# Description: Dockerfile for the API service
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8000

ENV FASTAPI_DEBUG=1
ENV DEBUG=1
ENV LOG_LEVEL=1
ENV TZ=Europe/Paris

CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-config=config/log_conf.yaml"]
