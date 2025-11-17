FROM apache/airflow:2.9.1-python3.10

USER root
RUN apt-get update && apt-get install -y git

USER airflow
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt