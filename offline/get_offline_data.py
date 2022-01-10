from datetime import datetime, timedelta
import pandas as pd

from feast import FeatureStore

# The entity dataframe is the dataframe we want to enrich with feature values
entity_df = pd.DataFrame.from_dict(
    {
        "id": ["A76", "A124", "A192"],
        "label": [-1, -1, 2], 
        "event_timestamp": [
            datetime.now() - timedelta(minutes=25),
            datetime.now() - timedelta(minutes=20),
            datetime.now() - timedelta(minutes=18),
        ],
        
    }
)


store = FeatureStore(repo_path=".")

#"select \"id\", \"event_timestamp\" as event_timestamp from \"Author\""
training_df = store.get_historical_features(
    entity_df=entity_df,
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
print(training_df.head())