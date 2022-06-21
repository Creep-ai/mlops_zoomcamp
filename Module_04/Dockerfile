FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN set -ex \
    && apt-get update \
    && pip install pipenv \
    && apt-get --yes autoremove \
    && apt-get --yes clean

WORKDIR /app
COPY . .

RUN pipenv install --system --deploy --ignore-pipfile

CMD python Module_04.py ./model.bin ./data/raw/ 2021 04 ./data/processed/