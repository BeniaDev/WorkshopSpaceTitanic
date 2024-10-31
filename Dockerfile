FROM python:3.10-slim

COPY requirements.txt /WORKSPACE/
RUN pip install -r /WORKSPACE/requirements.txt

COPY app/ ./WORKSPACE/app/
COPY data/ ./WORKSPACE/data/
COPY models/ ./WORKSPACE/models/