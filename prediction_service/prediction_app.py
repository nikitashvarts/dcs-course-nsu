import os
from typing import Dict

import joblib
from sklearn.pipeline import Pipeline


MODELS_DIR = './models'


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


def predict(text: str, model_name: str) -> Dict[str, float]:
    model_path = os.path.join(MODELS_DIR, model_name)

    assert os.path.exists(model_path), f'Model located at {model_path} does not exist'

    clf: Pipeline = load_model(model_path)

    probas = clf.predict_proba([text.lower()])[0]
    return {'Hate Speech': probas[0],
            'Offensive Language': probas[1],
            'Neither': probas[2]}
