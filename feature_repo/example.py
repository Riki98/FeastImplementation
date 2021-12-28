# This is an example feature definition file

from google.protobuf.duration_pb2 import Duration

from feast import Entity, Feature, FeatureView, ValueType

from neo4j_datasource import Neo4jSource

# Read data from parquet files. Parquet is convenient for local development mode. For
# production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# for more info.
authors_source_view = Neo4jSource(
    query="SELECT * FROM authors", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)


#for redis
""" driver_hourly_stats_redis = FileSource(
    path="/content/feature_repo/data/driver_stats.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
) """

# Define an entity for the driver. You can think of entity as a primary key used to
# fetch features.
# driver = Entity(name="driver_id", value_type=ValueType.INT64, description="driver id")
index_entity = Entity(name="index", value_type=ValueType.STRING, description="author id")


# Our parquet files contain sample data that includes a driver_id column, timestamps and
# three feature column. Here we define a Feature View that will allow us to serve this
# data to our model online.
authors_view_postgres = FeatureView(
    name="authors_view",
    entities=["index"],
    ttl=Duration(seconds=86400 * 1),
    features=[
        Feature(name="Authors_embedding", dtype=ValueType.FLOAT_LIST)
    ],
    online=True,
    batch_source=authors_source_view,
    tags={}
)


#redis
""" driver_hourly_stats_view_redis = FeatureView(
    name="driver_hourly_stats_redis",
    entities=["driver_id"],
    ttl=Duration(seconds=86400 * 1),
    features=[
        Feature(name="conv_rate", dtype=ValueType.FLOAT),
        Feature(name="acc_rate", dtype=ValueType.FLOAT),
        Feature(name="avg_daily_trips", dtype=ValueType.INT64),
    ],
    online=True,
    batch_source=driver_hourly_stats_redis,
    tags={},
)  """