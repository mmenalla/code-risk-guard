import pandas as pd
from pymongo import MongoClient
import logging

logger = logging.getLogger(__name__)


def push_to_mongo(df, mongo_uri: str, mongo_db: str, mongo_collection: str):
    """
    Appends a DataFrame to a MongoDB collection incrementally.
    """
    if df.empty:
        logger.warning("DataFrame is empty. Nothing to insert into MongoDB.")
        return

    try:
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    except Exception as e:
        logger.error(f"Error converting 'created_at' to datetime: {e}")

    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    collection = db[mongo_collection]

    last_date = None
    if collection.count_documents({}) > 0:
        try:
            last_date = collection.find_one(sort=[("created_at", -1)])["created_at"]
            last_date = pd.to_datetime(last_date)
            if last_date.tzinfo is None:
                last_date = last_date.tz_localize('UTC')
        except Exception as e:
            logger.error(f"Error fetching last processed date: {e}")
            last_date = None

    if last_date is not None:
        df = df[df['created_at'] > last_date]
        if df.empty:
            logger.info("No new records to insert. Incremental load skipped.")
            return

    records = df.to_dict(orient="records")
    result = collection.insert_many(records)
    logger.info(f"Inserted {len(result.inserted_ids)} records into MongoDB collection '{mongo_collection}'")

def load_labeled_data_from_mongo(
    mongo_uri: str,
    mongo_db: str,
    mongo_collection: str
) -> pd.DataFrame:
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


def log_model_predictions(
    predictions_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    model_name: str,
    mongo_uri: str = "mongodb://admin:admin@mongo:27017/risk_model_db?authSource=admin",
    db_name: str = "risk_model_db",
    collection_name: str = "model_predictions",
):
    """
    Logs model predictions together with features used for generation.
    Returns a mapping of {module: prediction_id} for downstream linking.
    """
    import pymongo
    from datetime import datetime

    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    merged = feature_df.merge(
        predictions_df[['module', 'risk_score', 'needs_maintenance']],
        on='module',
        how='inner'
    )

    records = []
    for _, row in merged.iterrows():
        features = {
            k: v for k, v in row.items()
            if k not in ['module', 'risk_score', 'needs_maintenance']
        }
        record = {
            "model_name": model_name,
            "module": row['module'],
            "features": features,
            "predicted_risk": float(row['risk_score']),
            "needs_maintenance": int(row['needs_maintenance']),
            "created_at": datetime.utcnow(),
            "source": "model_inference",
        }
        records.append(record)

    if not records:
        logging.warning("No prediction records to insert.")
        return {}

    result = collection.insert_many(records)

    id_mapping = {r['module']: str(_id) for r, _id in zip(records, result.inserted_ids)}

    logging.info(f"Inserted {len(records)} prediction records with features into '{collection_name}'.")
    return id_mapping


def log_human_feedback(
    module: str,
    repo_name: str,
    predicted_risk: float,
    manager_risk: float,
    prediction_id: str,
    user_id: str,
    mongo_uri: str = "mongodb://admin:admin@localhost:27017/risk_model_db?authSource=admin",
    db_name: str = "risk_model_db",
    collection_name: str = "risk_feedback",
):
    """
    Logs human (manager) feedback on model predictions to MongoDB.
    """
    import pymongo
    from datetime import datetime

    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    record = {
        "repo_name": repo_name,
        "module": module,
        "predicted_risk": float(predicted_risk),
        "manager_risk": float(manager_risk),
        "prediction_id": prediction_id,
        "user_id": user_id,
        "created_at": datetime.utcnow(),
        "source": "manual_feedback",
    }

    collection.insert_one(record)
    logging.info(f"✅ Logged feedback for {module} by {user_id} (pred={predicted_risk:.3f}, manager={manager_risk:.3f})")
