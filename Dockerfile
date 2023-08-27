FROM python:3.8-slim-buster

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt

# EXPOSE $PORT

# CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
CMD ["python3", "app.py"]