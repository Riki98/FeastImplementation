from datetime import datetime, timedelta
import pandas as pd

from feast import FeatureStore, Entity

# The entity dataframe is the dataframe we want to enrich with feature values


entity_auth = Entity("id_auth")

store = FeatureStore(repo_path="./feature_repo")

# {"index":["A76"], "event_timestamp":["2021-12-23 17:09:28.058854"], "created": ["2021-12-23 17:09:28.058854"]}
training_df = store.get_historical_features(
    entity_df="select \"index\", event_timestamp from authors",
    features=[
        "authors_view:Authors_embedding",
    ]
).to_df()

print("----- Feature schema -----\n")
print(training_df.info())

print()
print("----- Example features -----\n")
print(training_df.head())