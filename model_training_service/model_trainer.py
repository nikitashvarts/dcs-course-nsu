#!/usr/bin/env python3

import argparse
import os
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from stop_words import get_stop_words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Offensive Speech Classifier')
    parser.add_argument('--data-csv', type=str, required=True,
                        help='Path to file with train data.')
    parser.add_argument('--model-file', type=str, default=None,
                        help='File name for model to save after training.')
    return parser.parse_args()


def read_csv_data(path: str) -> pd.DataFrame:
    """
    Reads data from given CSV file and stores it in Pandas DataFrame.

    :param path: Path to file with comma-separated data.
    :return: Pandas DataFrame with data from the input file.
    """
    assert path.split('.')[-1] == 'csv', 'Input file must be of type CSV'
    assert os.path.exists(path), 'Input file does not exist'

    df: pd.DataFrame = pd.read_csv(path, usecols=['class', 'tweet'])

    print('[INFO] Data were successfully loaded from CSV.')
    return df


def prepare_text_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic text transformations for further text processing.

    :param df: Pandas DataFrame with text data to transform.
    :return: Pandas DataFrame with preprocessed text data.
    """
    df['tweet'] = df['tweet'].apply(lambda tweet: re.sub('[^A-Za-z]+', ' ', tweet.lower()))
    print('[INFO] Data were successfully preprocessed.')
    return df


def save_model(model: Pipeline, file_name: str, save_loc: str = '/models') -> None:
    os.makedirs(save_loc, exist_ok=True)
    joblib.dump(model, os.path.join(save_loc, file_name))
    print(f'Trained model was successfully saved to {save_loc}')


if __name__ == '__main__':
    args: argparse.Namespace = parse_args()

    print('[INFO] Started model training.')

    data_file: str = args.data_csv
    model_file: str = args.model_file

    data_df: pd.DataFrame = read_csv_data(path=data_file)

    data_df = data_df.iloc[:1001]

    clf: Pipeline = Pipeline(
        steps=[
            ('Vectorizer', TfidfVectorizer(stop_words=get_stop_words('en'))),
            ('Classifier', OneVsRestClassifier(SVC(kernel='linear', probability=True)))
        ],
        verbose=True,
    )

    clf = clf.fit(X=data_df['tweet'], y=data_df['class'])

    print('Classifier was successfully trained.')

    save_model(model=clf, file_name=model_file)
