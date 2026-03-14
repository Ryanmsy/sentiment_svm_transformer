Notes: I removed our saved model (pkl file and bert model) because it was too large. Will find a solution.

## Project Structure (Claude Code)


sentiment_svm_transformer/
├── data/
│   ├── amazon_test_2500.xlsx       # Source data (2,500 Amazon reviews)
│   ├── amazon_reviews.db           # SQLite DB for transformer pipeline
│   └── corporate_data_warehouse.db # SQLite DB for SVM pipeline
├── app/
│   ├── config.py                   # Path/env config (DB paths, model dirs)
│   ├── transformer_sentiment.py    # DistilBERT: full train + evaluate pipeline
│   ├── transformer_predict.py      # DistilBERT: inference-only on customer data
│   ├── svm_sentiment.py            # TF-IDF + LinearSVC train + evaluate pipeline
│   ├── main_sentiment.py           # Streamlit UI (model selection + predictions)
│   └── database_sentimentanalysis.py # ETL: Excel → SQLite
└── pyproject.toml




## What would we do differently? - Scaling at enterprise level

### 1. Distributed Data Processing
* Issue: Our initial prototype relied on Pandas, which processes data in memory on a single machine. At a large scale like Walmart, this would cause memory errors and slow performance because the entire dataset must fit into a single machine.


* Future Improvement:
We would replace Pandas with Apache Spark. Spark uses a "MapReduce" feature to split the dataset into chunks and process them in parallel across a cluster of machines.


### 2. Container Orchestration (Kubernetes) 
* Issue: Docker can containerized our application, but if there are multiple containers, it can be insufficient or difficult to manage. If the single container crashes or receives 10000 requests per second, the container will break.

* Future Improvement:
Kubernetes: We would deploy our Docker containers into Kubernetes cluster. It will manages the containers.



### 3.Containerization Strategy 
* Issue: Compatibility issues with CPU architectures between OS (ARM Macs and x86-based Windows machines).

* Future Improvement:
We would implement multiple containers using docker buildx. This allows us to push a single image tag that works on both ARM64 and AMD64 hardware to solve "it works on my machine" problem.


### 4. Implementation of MLOps Pipeline
* Issue: Our workflow has no separation between  training data and  inference code.

* Future Improvement:
We would establish a CI/CD pipeline:

Data Stage (Spark): Raw data is processed and features are stored in a Feature Store.

Training (Dev): Model is trained and artifacts are versioned.

Inference: Only the saved model weights and scoring script are deployed to the production cluster.
