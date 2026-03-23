# python image
FROM python:3.12-slim

# Avoid .pyc file and __pycahce__ folder
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create contener root folder
WORKDIR /app

# Copy requirements files and config into the workdir (COPY <source1> <source2> <source3> ... <destination>)
COPY requirements.txt requirements-airflow.txt pyproject.toml README.md ./
COPY src/ ./src/

# Install pyproject and requirements
RUN python -m pip install --upgrade pip
RUN python -m pip install -e .
RUN python -m pip install -r requirements-airflow.txt --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.1.8/constraints-3.12.txt"
# For use CURL
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY api_service/ ./api_service/
COPY streamlit_app/ ./streamlit_app/
COPY scripts/ ./scripts/
COPY airflow/ ./airflow/

ENV PYTHONPATH=/app/src
