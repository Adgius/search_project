import torch
import numpy as np
import redis
import logging 
import json

from elasticsearch import Elasticsearch
from kafka import KafkaConsumer
from utils import (
    MLService,
    CandidateModel,
    collate_fn,
    RankingDataset
)

logging.basicConfig(level=logging.INFO)


def pipeline(query):
    global candidate_model
    global ml_service
    global ml_exists

    candidates = candidate_model.query(query, size=20)
    candidates = [i[1] for i in candidates]
    if ml_exists:
        ds = RankingDataset(query=query,
                    candidates=candidates,
                    vocab=ml_service.vocab,
                    oov_val=ml_service.vocab['OOV'],
                    preproc_func=ml_service.simple_preproc)
        dl = torch.utils.data.DataLoader(
            ds, 
            batch_size=ml_service.dataloader_bs, 
            num_workers=0,
            collate_fn=collate_fn, 
            shuffle=False)
    
        score = ml_service.predict(dl)
        return {'result': list(np.array(candidates)[np.argsort(score)])}
    else:
        return {'result': candidates}

def process_message(message):
    global redis

    request_id = message.value['index']
    query = message.value['question']

    result = pipeline(query)

    # Save the results to Redis and publish to the Pub/Sub channel
    result_data = json.dumps(result)
    redis.set(request_id, result_data, ex=300)  # TTL = 5 minutes
    redis.publish(request_id, result_data)  # Publish the result to the channel


if __name__ == "__main__":
    # Elasticsearch setup
    es = Elasticsearch(hosts=["http://elasticsearch:9200"])

    # Redis connection
    redis = redis.from_url("redis://redis:6379")

    # Kafka connection
    kafka_consumer = KafkaConsumer(
        'search_terms',
        bootstrap_servers=['kafka:29092'],
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        group_id="search_service_group"
    )

    # Candidate model setup
    index_name = 'question_base'
    candidate_model = CandidateModel(es, index_name)

    # ML model setup
    ml_exists = True
    try:
        ml_service = MLService()
    except BaseException as e:
        logging.error(f'Cannot setup ML service. {e}')
        ml_exists = False

    for message in kafka_consumer:
        process_message(message)