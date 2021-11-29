from pandas import pandas as pd
from datetime import datetime
from pprint import pprint
from feast import FeatureStore

store = FeatureStore(repo_path=".")

feature_vector = store.get_online_features(
    features=[
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate",
        "driver_hourly_stats:avg_daily_trips",
    ],
    entity_rows=[
        {"driver_id": 1004},
        {"driver_id": 1005},
    ],
).to_dict()

pprint(feature_vector)

pd.read_parquet("feast_repo/data/driver_stats.parquet") #anche con la demo è vuoto, vedere come riempire
print("-------------------------")