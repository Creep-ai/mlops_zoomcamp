import os
import sys
import pickle

import pandas as pd


def get_input_path(year: int, month: int) -> str:
    default_input_pattern = (
        'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/'
        'fhv/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    )
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year: int, month: int) -> str:
    default_output_pattern = './taxi_type=fhv_year={year:04d}_month={month:02d}.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(filename: str) -> pd.DataFrame:
    options = {'client_kwargs': {'endpoint_url': 'http://localhost:4566'}}

    df = pd.read_parquet(filename, storage_options=options)
    return df


def save_data(data: pd.DataFrame, filename: str) -> None:
    options = {'client_kwargs': {'endpoint_url': 'http://localhost:4566'}}
    data.to_parquet(filename, engine='pyarrow', compression=None, index=False, storage_options=options)


def prepare_data(df: pd.DataFrame, categorical: list) -> pd.DataFrame:
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = round(df.duration.dt.total_seconds() / 60, 2)

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


def main(year: int, month: int) -> None:
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    categorical = ['PUlocationID', 'DOlocationID']
    df = read_data(input_file)
    df = prepare_data(df, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())
    print('predicted sum duration:', y_pred.sum())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(df_result, output_file)


if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year=year, month=month)
