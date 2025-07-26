FROM python:3.11-slim

WORKDIR /srv/app

COPY . .

RUN apt update && \
    apt upgrade -yq

RUN pip install poetry

RUN poetry install --no-root
