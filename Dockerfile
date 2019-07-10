FROM python:3.7-slim

ENV HOME /root

COPY . $HOME

WORKDIR $HOME

RUN apt-get update
RUN apt-get install -y libgomp1
RUN pip install .
