from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from configs.config_loader import config

mongo_url = config["mongodb"]["url"]
db_name = config["mongodb"]["database"]

client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
db = client[db_name]

drivers_collection = db["drivers"]
sessions_collection = db["sessions"]
alerts_collection = db["alerts"]
daily_scores_collection = db["daily_scores"]

# Verify connection and database (fail fast on startup if unreachable)
try:
    client.admin.command("ping")
    # Ensure we can list collections (confirms DB access)
    db.list_collection_names()
    print(f"MongoDB connected: database={db_name}")
except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
    print(f"MongoDB connection failed: {e}")
    raise