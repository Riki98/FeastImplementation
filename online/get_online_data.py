# docker run --name redis --rm -p 6379:6379 -d redis

from datetime import datetime, timedelta
import pandas as pd

from feast import FeatureStore, Entity, ValueType

# The entity dataframe is the dataframe we want to enrich with feature values


index_entity_auth = Entity(name="index_auth", value_type=ValueType.STRING, description="author id", join_key="id")
#index_entity_paper = Entity(name="index_paper", value_type=ValueType.STRING, description="paper id", join_key="id")

store = FeatureStore(repo_path="./")

# questo diventa il modello di train
training_df = store.get_online_features(
    features=[
        "authors_view:graphsage_embedding",
        "authors_view:fastrp_embedding",
    ],
    entity_rows=[
        {"index_auth": "A76"},
        {"index_auth": "A124"}
    ]
).to_df()


print("----- Feature schema -----\n")
print(training_df.info())

print()
print("----- Example features -----\n")
print(training_df.head())