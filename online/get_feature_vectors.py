from pprint import pprint
from feast import FeatureStore

store = FeatureStore(repo_path=".")

feature_vector = store.get_online_features(
    features=[
        "driver_hourly_stats_postgres:conv_rate",
        "driver_hourly_stats_postgres:acc_rate",
        "driver_hourly_stats_postgres:avg_daily_trips",
    ],
    entity_rows=[
        {"driver_id": 1004},
        {"driver_id": 1005},
    ],
).to_dict()



pprint(feature_vector)