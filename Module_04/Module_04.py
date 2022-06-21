import pickle
import sys
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


def load_model(model_path: str) -> Tuple[DictVectorizer, LinearRegression]:
    with open(model_path, 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr


def read_prepare_data(filename: str, categorical: list) -> Tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(filename)
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    dicts = df[categorical].to_dict(orient='records')
    return df, dicts


def ride_duration_prediction(model_path: str, raw_data_path: str,
                             year: str, month: str, processed_data_path: str) -> None:
    input_file = raw_data_path + f'fhv_tripdata_{year}-{month}.parquet'
    output_data_path = processed_data_path + f'fhv_tripdata_{year}-{month}_predictions.parquet'

    categorical = ['PUlocationID', 'DOlocationID']

    dv, lr = load_model(model_path)
    df, dicts = read_prepare_data(input_file, categorical)
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print(f'Mean prediction: {y_pred.mean()}')
    df['ride_id'] = f'{year:.04}/{month:.02}_' + df.index.astype('str')
    df['predictions'] = y_pred

    df[['ride_id', 'predictions']].to_parquet(
        output_data_path,
        engine='pyarrow',
        compression=None,
        index=False
    )


def run() -> None:
    model_path = sys.argv[1]  # './models/model.bin'
    raw_data_path = sys.argv[2]  # './data/raw/'
    year = sys.argv[3]  # '2021'
    month = sys.argv[4]  # '03'
    processed_data_path = sys.argv[5]  # './data/processed/'

    ride_duration_prediction(model_path=model_path,
                             raw_data_path=raw_data_path,
                             year=year,
                             month=month,
                             processed_data_path=processed_data_path)


if __name__ == '__main__':
    run()
