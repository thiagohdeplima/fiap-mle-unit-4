FROM python:3.12-slim

WORKDIR /srv/app

COPY . .

RUN apt update && \
    apt upgrade -yq

RUN pip install poetry

RUN postry install --no-root
