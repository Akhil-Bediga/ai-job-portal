from functools import lru_cache

from pymongo import MongoClient
from pymongo.database import Database

from src.config import mongo_cfg


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    # short timeout so a missing Mongo fails fast with a clear error
    return MongoClient(mongo_cfg.uri, serverSelectionTimeoutMS=5000)


def get_db() -> Database:
    return get_client()[mongo_cfg.database]
