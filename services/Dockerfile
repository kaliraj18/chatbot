FROM python:3.10.11

WORKDIR /flask-app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "waitress-serve", "--port", "8081", "--call", "main.app:create_app"]
