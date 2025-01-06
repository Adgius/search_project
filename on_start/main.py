import io
import requests
import polars as pl
import numpy as np
import time

from zipfile import ZipFile
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from kafka import KafkaProducer
from tqdm import tqdm

import json
import uuid
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

index_name = "question_base"

kafka_config = {
        'bootstrap_servers': 'kafka:29092',
        'client_id': 'search_terms_client',
        'value_serializer': lambda obj: json.dumps(obj).encode('utf-8'),
    }

def do_while_success(func):
    def wrapper(**args):
        not_success = True
        while not_success:
            try:
                output = func(**args)
                not_success = False
            except Exception as e:
                logging.error(e)
                time.sleep(5)
        return output
    return wrapper
            
@do_while_success
def create_index(index_name):
    es = Elasticsearch(["http://elasticsearch:9200"])

    mapping = {
        "mappings": {
            "properties": {
                "question": {
                    "type": "text",
                    "analyzer": "standard", 
                    "search_analyzer": "english" 
                },
                "index": {
                    "type": "keyword"
                }
            }
        }
    }

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)

@do_while_success
def load_data():
    resp = requests.get("https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip")

    arch = ZipFile(io.BytesIO(resp.content))

    schema = {'id': pl.Int64, 
            'id_left': pl.Int64, 
            'id_right': pl.Int64, 
            'text_left': pl.String, 
            'text_right': pl.String, 
            'label': pl.Int8}

    def read_tsv(path, schema):
        data = []
        first = True
        with arch.open(path) as zp:
            for line in zp.readlines():
                if first:
                    first = False
                else:
                    data.append(line.decode('utf-8').strip().split("\t"))
            data = pl.DataFrame(data=data, schema=schema)
        return data

    train = read_tsv('QQP/train.tsv', schema)
    test = read_tsv('QQP/dev.tsv', schema)

    idx_df = pl.concat([test.unique('id_left').select(pl.col('id_left'), pl.col('text_left')).rename({'id_left': 'idx', 'text_left': 'text'}),
                        test.unique('id_right').select(pl.col('id_right'), pl.col('text_right')).rename({'id_right': 'idx', 'text_right': 'text'}),
                        train.unique('id_left').select(pl.col('id_left'), pl.col('text_left')).rename({'id_left': 'idx', 'text_left': 'text'}),
                        train.unique('id_right').select(pl.col('id_right'), pl.col('text_right')).rename({'id_right': 'idx', 'text_right': 'text'})
            ]).unique()
    return idx_df

@do_while_success
def send_to_kafka(config, data: pl.DataFrame):
    producer = KafkaProducer(**config)
    for idx, query in tqdm(data.iter_rows()):
        producer.send(topic=index_name,
                    value={'index': str(idx),
                            'question': query}
                    )
        
if __name__ == "__main__":
    logging.info('Start pipeline ...')
    create_index(index_name=index_name)
    logging.info('Index created')
    data = load_data()
    logging.info('Data loaded')
    send_to_kafka(config=kafka_config, data=data)
    logging.info('Successfully completed')
