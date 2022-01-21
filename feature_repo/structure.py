from os import path
from google.protobuf.duration_pb2 import Duration
from feast import Entity, Feature, FeatureView, ValueType
from feast_postgres import PostgreSQLSource


#creo le view da postgres
authors_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"Author\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)

# Define an entity for the auth
index_entity_auth = Entity(name="index_auth", value_type=ValueType.STRING, description="author id", join_key="id")

# postgres e redis usano le stesse identiche feature view
author_embedding_view_postgres = FeatureView(
    name="authors_view",
    entities=["index_auth"],
    ttl=Duration(seconds=8000 * 1),
    features=[
        Feature(name="name", dtype=ValueType.STRING),
        Feature(name="graphsage_embedding", dtype=ValueType.FLOAT_LIST),
        Feature(name="fastrp_embedding", dtype=ValueType.FLOAT_LIST)
    ],
    online=True,
    batch_source=authors_source_view_postgres,
    tags={}
)



######################### PAPER

#creo le view da postgres
papers_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"Paper\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)

# Define an entity for the auth
index_entity_paper = Entity(name="index_paper", value_type=ValueType.STRING, description="paper id", join_key="id")

# postgres e redis usano le stesse identiche feature view
paper_embedding_view_postgres = FeatureView(
    name="papers_view",
    entities=["index_paper"],
    ttl=Duration(seconds=8000 * 1),
    features=[
        Feature(name="name", dtype=ValueType.STRING),
        Feature(name="graphsage_embedding", dtype=ValueType.FLOAT_LIST),
        Feature(name="fastrp_embedding", dtype=ValueType.FLOAT_LIST)
    ],
    online=True,
    batch_source=papers_source_view_postgres,
    tags={}
)

