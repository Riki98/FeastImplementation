from os import path
from google.protobuf.duration_pb2 import Duration
from feast import Entity, Feature, FeatureView, ValueType
from feast_postgres import PostgreSQLSource
import neo4j_datasource

# Inserisco i dati da neo4j a postgres

df_auth = neo4j_datasource.run_retrieve_neo4j_node("Author")
#neo4j_datasource.run_create_offline_table("Author", df_auth, "replace")
neo4j_datasource.run_store_data("Author", df_auth)
#neo4j_datasource.run_drop_offline_table("Author")

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
    ttl=Duration(seconds=86400 * 1),
    features=[
        Feature(name="name", dtype=ValueType.STRING),
        Feature(name="graphsage_embedding", dtype=ValueType.STRING),
        Feature(name="fastrp_embedding", dtype=ValueType.STRING)
    ],
    online=True,
    batch_source=authors_source_view_postgres,
    tags={}
)
