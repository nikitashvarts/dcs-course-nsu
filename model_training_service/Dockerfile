FROM python:3.6

RUN pip install pandas==1.1.5 scikit-learn==0.24.1 stop-words==2018.7.23

COPY . .

CMD python model_trainer.py --data-csv '/data/speech.csv' --model-file 'hate_speech_model'
