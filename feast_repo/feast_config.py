import feast
from pandas import pandas as pd
from datetime import datetime
from pprint import pprint
from feast import FeatureStore
import spark

store = FeatureStore(repo_path=".")

feature_vector = store.get_online_features(
    features=[
        {"driver_hourly_stats:conv_rate": 5},
        {"driver_hourly_stats:acc_rate": 6},
        {"driver_hourly_stats:avg_daily_trips": 7},
    ],
    entity_rows=[
        {"driver_id": 1004},
        {"driver_id": 1005},
    ],
).to_dict()

pprint(feature_vector)
print("\n\n")
print(feature_vector)
print("\n\n")

parquetFile = spark.read.parquet("/data/driver_stats.parquet")
query = f"SELECT * FROM {parquetFile}"
test = spark.sql(query)
test.show()

print("-------------------------")