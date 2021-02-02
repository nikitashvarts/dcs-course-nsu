FROM python:3.6

RUN pip install scikit-learn==0.24.1 firefly-python==0.1.15

COPY . .

CMD firefly prediction_app.predict --bind 0.0.0.0:5000

EXPOSE 5000
