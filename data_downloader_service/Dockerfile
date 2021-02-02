FROM python:3.6

COPY . .

CMD python data_downloader.py --data-url 'https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv' --save-loc '/data/speech.csv'