import uuid
import json
import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from kafka import KafkaProducer
from langdetect import detect
from redis import asyncio as aioredis

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global producer
    global redis
    global is_initialized 
    is_initialized = False
    # Load the ML model
    config = {
        'bootstrap_servers': 'kafka:29092',
        'client_id': 'search_terms_client',
        'value_serializer': lambda obj: json.dumps(obj).encode('utf-8'),
    }
    producer = KafkaProducer(**config)
    redis = aioredis.from_url("redis://redis:6379")
    is_initialized = True
    yield
    redis.close()
    producer.close()

app = FastAPI(lifespan=lifespan)

@app.get("/ping")
def ping():
    global is_initialized
    if is_initialized:
        return JSONResponse(status_code=200, content={"status": "ok"})
    return JSONResponse(status_code=200, content={"status": "initializing"})

@app.post("/search/{query}")
async def search(query: str):
    lang = detect(query)
    if lang != 'en':
        return JSONResponse(status_code=400, content={"message": "Language is invalid"})
    request_id = str(uuid.uuid4()) 
    producer.send(topic='search_terms', value={'request_id': request_id, 
                                               'question': query}
             )
    return JSONResponse(status_code=200, content={"request_id": request_id})


@app.get("/result/{request_id}")
async def get_result(request_id: str, query: str):
    global redis
    async def event_stream():
        result = await redis.get(query)
        if result:
            yield f"data: {result.decode()}\n\n"
            return
        pubsub = redis.pubsub()
        await pubsub.subscribe(request_id)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield f"data: {message['data'].decode()}\n\n"
                    break
        finally:
            await pubsub.unsubscribe(request_id)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
