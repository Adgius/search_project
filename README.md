# Kafka-Based ML Search Service

## Overview
This project is a distributed system designed to handle search queries using Apache Kafka, Elasticsearch, and an ML-based ranking service. The architecture is containerized using Docker and orchestrated with Docker Compose.

## Architecture
![image](https://github.com/user-attachments/assets/2bbbbfbe-57d3-42ab-80fb-90a16b44dbfa)

### Key Components:
1. **Kafka Cluster**
    - **Zookeeper**: Manages Kafka cluster metadata.
    - **Kafka Broker**: Handles message production and consumption.
    - **Kafdrop**: Provides a UI to monitor Kafka topics.
    - **Kafka Exporter**: Exports Kafka metrics for Prometheus.
    - **Kafka Connect**: Facilitates data integration with Elasticsearch.
    - **Kafka Connect UI**: UI for managing Kafka connectors.

2. **Elasticsearch and Kibana**
    - **Elasticsearch**: Full-text search engine.
    - **Kibana**: Visualization tool for Elasticsearch data.

3. **Redis**
    - **Redis**: Cache and Pub/Sub messaging.
    - **RedisInsight**: UI for managing Redis.

4. **ML Service**
    - Ranks search results based on an ML model.
    - Handles feature extraction and prediction.

5. **Query Handler**
    - FastAPI service for handling user queries.
    - Publishes queries to Kafka and fetches results from Redis.

6. **Monitoring Tools**
    - **Prometheus**: Monitors system metrics.
    - **Grafana**: Visualizes Prometheus metrics.

7. **On-Start Service**
    - Initializes the system by loading data into Elasticsearch and Kafka.

## Getting Started

### Prerequisites
- Docker
- Docker Compose

### Project Structure
```
.
├── docker-compose.yml
├── on_start
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
├── query_handler
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
├── ml_service
│   ├── Dockerfile
│   ├── main.py
|   ├── model.py
│   ├── utils.py
│   ├── requirements.txt
├── prometheus
│   ├── prometheus.yml
├── grafana
│   ├── datasources.yml
│   ├── dashboards.yml
│   ├── dashboards
│       ├── elasticsearch-dashboard.json
|       ├── kafka-dashboard.json
```

### Installation and Deployment
1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2. Build and start the services:
    ```bash
    docker-compose up --build
    ```
3. Access services:
    - Kafka UI: [http://localhost:9000](http://localhost:9000)
    - Elasticsearch: [http://localhost:9200](http://localhost:9200)
    - Kibana: [http://localhost:5601](http://localhost:5601)
    - Redis UI: [http://localhost:5540](http://localhost:5540)
    - Prometheus: [http://localhost:9090](http://localhost:9090)
    - Grafana: [http://localhost:3000](http://localhost:3000)

### Usage

#### 1. On-Start Service
- Initializes Elasticsearch with the `question_base` index.
- Downloads and loads the QQP dataset into Kafka.

#### 2. Query Handler API
- Start querying the service:
    - **Health Check:**
      ```
      GET /ping
      ```
    - **Submit a Search Query:**
      ```
      POST /search/{query}
      ```
      Example:
      ```bash
      curl -X POST http://localhost:5000/search/<your-example-question>
      ```
    - **Fetch Search Results:**
      ```
      GET /result/{request_id}
      ```
      Example:
      ```bash
      curl -X GET http://localhost:5000/result/<question-id-from-search-request>?query=<your-example-question>
      ```

### Environment Variables

#### ML Service:
- `EMB_PATH_GLOVE`: Path to GloVe embeddings.
- `EMB_PATH_KNRM`: Path to KNRM embeddings.
- `MLP_PATH`: Path to the MLP model.
- `VOCAB_PATH`: Path to the vocabulary JSON.

### Data Flow
1. **Data Initialization**: The `on_start` service downloads the QQP dataset, processes it, and loads it into Kafka and Elasticsearch.
2. **Query Handling**: User queries are handled by the `query_handler`, which publishes them to Kafka.
3. **Search Results**:
    - The `ml_service` consumes queries from Kafka, performs ranking, and stores the results in Redis.
    - Results are published to Redis channels for real-time updates.

### Monitoring
- Configure Prometheus with the provided `prometheus.yml`.
- Visualize metrics in Grafana dashboards.

### Contributing
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request.

