import pandas as pd
from pymongo import MongoClient
import logging

logger = logging.getLogger(__name__)


def push_to_mongo(df, mongo_uri: str, mongo_db: str, mongo_collection: str):
    """
    Appends a DataFrame to a MongoDB collection.

    Args:
        df (pd.DataFrame): Data to insert.
        mongo_uri (str): MongoDB connection URI.
        mongo_db (str): Database name.
        mongo_collection (str): Collection name.
    """
    if df.empty:
        logger.warning("DataFrame is empty. Nothing to insert into MongoDB.")
        return

    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    collection = db[mongo_collection]

    records = df.to_dict(orient="records")
    result = collection.insert_many(records)
    logger.info(f"Inserted {len(result.inserted_ids)} records into MongoDB collection '{mongo_collection}'")

def load_labeled_data_from_mongo(
    mongo_uri: str,
    mongo_db: str,
    mongo_collection: str
) -> pd.DataFrame:
    """Fetch labeled data from MongoDB and return as a Pandas DataFrame"""
    logger.info(f"Connecting to MongoDB at {mongo_uri}...")
    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    collection = db[mongo_collection]

    records = list(collection.find({}))
    if not records:
        logger.warning(f"No records found in collection '{mongo_collection}'.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "_id" in df.columns:
        df.drop(columns=["_id", "created_at"], inplace=True)

    logger.info(f"Loaded {len(df)} rows from MongoDB collection '{mongo_collection}'.")
    return df

def get_last_processed_date(mongo_uri, mongo_db, mongo_collection):
    from pymongo import MongoClient
    import pandas as pd

    client = MongoClient(mongo_uri)
    collection = client[mongo_db][mongo_collection]

    if collection.count_documents({}) == 0:
        return None

    last_date = collection.find_one(sort=[("created_at", -1)])["created_at"]

    return pd.to_datetime(last_date)


def log_model_metrics(metrics: dict, model_name: str = "xgboost_risk_model",
                      mongo_uri: str = "mongodb://admin:admin@mongo:27017/risk_model_db?authSource=admin",
                      db_name: str = "risk_model_db",
                      collection_name: str = "model_metrics"):
    import pymongo
    from datetime import datetime

    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    record = {
        "model_name": model_name,
        "accuracy": float(metrics.get("accuracy")),
        "tn": int(metrics.get("tn")),
        "fp": int(metrics.get("fp")),
        "fn": int(metrics.get("fn")),
        "tp": int(metrics.get("tp")),
        "timestamp": datetime.utcnow()
    }

    collection.insert_one(record)
    logger.info(f"Logged metrics for model '{model_name}' at {record['timestamp']}")
