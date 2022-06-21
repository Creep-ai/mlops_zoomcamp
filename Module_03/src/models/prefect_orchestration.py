from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle
from typing import Optional, Tuple

import pandas as pd
from prefect import flow, task
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.logging import get_run_logger
from prefect.orion.schemas.schedules import CronSchedule
from prefect.task_runners import SequentialTaskRunner
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


@task
def get_paths(date: Optional[str]) -> Tuple[str, str]:
    if date is None:
        train_month = str((datetime.now() - relativedelta(months=2)).month).zfill(2)
        val_month = str((datetime.now() - relativedelta(months=1)).month).zfill(2)
    else:
        train_month = str((datetime.strptime(date, '%Y-%m-%d') - relativedelta(months=2)).month).zfill(2)
        val_month = str((datetime.strptime(date, '%Y-%m-%d') - relativedelta(months=1)).month).zfill(2)
    train_path = f'../../data/raw/fhv_tripdata_2021-{train_month}.parquet'
    val_path = f'../../data/raw/fhv_tripdata_2021-{val_month}.parquet'
    return train_path, val_path


@task
def read_data(path: str) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info('read data')
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df: pd.DataFrame, categorical: list, train: bool = True) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info('prepare features')
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df: pd.DataFrame, categorical: list) -> Tuple[LinearRegression, DictVectorizer]:
    logger = get_run_logger()
    logger.info('start train')
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df: pd.DataFrame, categorical: list, dv: DictVectorizer, lr: LinearRegression) -> None:
    logger = get_run_logger()
    logger.info('run model')
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values
    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")


@flow(task_runner=SequentialTaskRunner())
def main(date: Optional[str] = None) -> None:
    train_path, val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    with open(f"../../models/model-{date or datetime.strptime(date, '%Y-%m-%d')}.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open(f"../../models/dv-{date or datetime.strptime(date, '%Y-%m-%d')}.pkl", "wb") as f:
        pickle.dump(dv, f)
    run_model(df_val_processed, categorical, dv, lr)


DeploymentSpec(
    flow=main,
    name="model_training",
    flow_runner=SubprocessFlowRunner(),
    tags=['fhv'],
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="Asia/Yekaterinburg")
)
