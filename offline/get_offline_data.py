""" from feast import FeatureStore


# A FeatureStore object is used to define, create, and retrieve features.
# repo_path (optional): Path to a `feature_store.yaml` used to configure the 
# feature store.
store = FeatureStore(repo_path=".")

query = "select \"id(n)\", \"event_timestamp\" as event_timestamp from \"Author\""
training_df = store.get_historical_features(
    entity_df=query,
    features=[
        "authors_view:name",
        "authors_view:graphsage_embedding",
        "authors_view:fastrp_embedding",
    ],
).to_df()

print("----- Feature schema -----\n")
print(training_df.info())

print()
print("----- Example features -----\n")
print(training_df.head()) """