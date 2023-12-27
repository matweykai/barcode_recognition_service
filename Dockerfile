FROM python:3.10

WORKDIR /app

RUN apt-get update && \
    apt-get install -y ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY weights weights
COPY setup.cfg setup.cfg
COPY tests tests
COPY src src
COPY Makefile Makefile
COPY app.py app.py
COPY config config

CMD python app.py
