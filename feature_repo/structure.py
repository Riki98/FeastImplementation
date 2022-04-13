from datetime import timedelta
from os import path
from google.protobuf.duration_pb2 import Duration
from feast import Entity, Feature, FeatureView, ValueType
from feast_postgres import PostgreSQLSource

################################################## AUTHOR  

# Source view for querying author table
authors_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"Author\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)

# Define an entity for the auth
index_entity_auth = Entity(name="author_id_neo4j", value_type=ValueType.INT64, description="author id neo4j")


# Defining feature view for querying paper table
author_embedding_view_postgres = FeatureView(
    name="author_view",
    entities=["author_id_neo4j"],
    ttl=timedelta(days=30),
    features=[
        Feature(name="name_author", dtype=ValueType.STRING),
        Feature(name="label_author", dtype=ValueType.FLOAT),
        Feature(name="author_id_neo4j", dtype=ValueType.INT64),
        Feature(name="graphsage_embedding_author", dtype=ValueType.FLOAT_LIST),
        Feature(name="fastrp_embedding_author", dtype=ValueType.FLOAT_LIST)
    ],
    online=True,
    batch_source=authors_source_view_postgres,
    tags={}
)


################################################## PAPER

# Source view for querying paper table
papers_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"Paper\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)

# Define an entity for the auth
index_entity_paper = Entity(name="paper_id_neo4j", value_type=ValueType.INT64, description="paper id neo4j")


# Defining feature view for querying paper table
paper_embedding_view_postgres = FeatureView(
    name="paper_view",
    entities=["paper_id_neo4j"],
    ttl=timedelta(days=30),
    features=[
        Feature(name="name_paper", dtype=ValueType.STRING),
        Feature(name="label_paper", dtype=ValueType.FLOAT),
        Feature(name="paper_id_neo4j", dtype=ValueType.INT64),
        Feature(name="graphsage_embedding_paper", dtype=ValueType.FLOAT_LIST),
        Feature(name="fastrp_embedding_paper", dtype=ValueType.FLOAT_LIST)
    ],
    online=True,
    batch_source=papers_source_view_postgres,
    tags={}
)


################################################## CONFERENCE

# Source view for querying conference table
conference_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"Conference\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)

# Define an entity for the conference
index_entity_conf = Entity(name="conference_id_neo4j", value_type=ValueType.INT64, description="conference id neo4j")


# Defining feature view for querying paper table
conference_embedding_view_postgres = FeatureView(
    name="conference_view",
    entities=["conference_id_neo4j"],
    ttl=timedelta(days=30),
    features=[
        Feature(name="name_conference", dtype=ValueType.STRING),
        Feature(name="label_conference", dtype=ValueType.FLOAT),
        Feature(name="conference_id_neo4j", dtype=ValueType.INT64),
        Feature(name="graphsage_embedding_conference", dtype=ValueType.FLOAT_LIST),
        Feature(name="fastrp_embedding_conference", dtype=ValueType.FLOAT_LIST)
    ],
    online=True,
    batch_source=conference_source_view_postgres,
    tags={}
)


################################################## TERM

# Source view for querying term table
term_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"Term\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)

# Define an entity for the auth
index_entity_term = Entity(name="term_id_neo4j", value_type=ValueType.INT64, description="term id neo4j")


# Defining feature view for querying paper table
term_embedding_view_postgres = FeatureView(
    name="term_view",
    entities=["term_id_neo4j"],
    ttl=timedelta(days=30),
    features=[
        Feature(name="term_id_neo4j", dtype=ValueType.INT64),
        Feature(name="graphsage_embedding_term", dtype=ValueType.FLOAT_LIST),
        Feature(name="fastrp_embedding_term", dtype=ValueType.FLOAT_LIST)
    ],
    online=True,
    batch_source=term_source_view_postgres,
    tags={}
)


################################################## PA

# Source view for querying pa table
PA_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"PA\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)


# Defining feature view for querying paper table
PA_embedding_view_postgres = FeatureView(
    name="pa_view",
    entities=["paper_id_neo4j", "author_id_neo4j"],
    ttl=timedelta(days=30),
    features=[
        Feature(name="name_pa", dtype=ValueType.STRING),
        Feature(name="label(n)_pa", dtype=ValueType.STRING),
        Feature(name="label(m)_pa", dtype=ValueType.STRING),
        Feature(name="pa_id_neo4j", dtype=ValueType.INT64),
        Feature(name="label_n_pa", dtype=ValueType.FLOAT)
    ],
    online=True,
    batch_source=PA_source_view_postgres,
    tags={}
)


################################################## PC

# Source view for querying pc table
PC_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"PC\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)


# Defining feature view for querying paper table
PC_embedding_view_postgres = FeatureView(
    name="pc_view",
    entities=["paper_id_neo4j", "conference_id_neo4j"],
    ttl=timedelta(days=30),
    features=[
        Feature(name="pc_id_neo4j", dtype=ValueType.INT64),
        Feature(name="name_PC", dtype=ValueType.STRING),
        Feature(name="label(n)_pc", dtype=ValueType.STRING),
        Feature(name="label(m)_pc", dtype=ValueType.STRING),
        Feature(name="label_n_pc", dtype=ValueType.FLOAT)
    ],
    online=True,
    batch_source=PC_source_view_postgres,
    tags={}
)


################################################## PT

# Source view for querying pt table
PT_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"PT\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)


# Defining feature view for querying paper table
PT_embedding_view_postgres = FeatureView(
    name="pt_view",
    entities=["paper_id_neo4j", "term_id_neo4j"],
    ttl=timedelta(days=30),
    features=[
        Feature(name="pt_id_neo4j", dtype=ValueType.INT64),
        Feature(name="name_pt", dtype=ValueType.STRING),
        Feature(name="label(n)_pt", dtype=ValueType.STRING),
        Feature(name="label(m)_pt", dtype=ValueType.STRING),
        Feature(name="label_n_pt", dtype=ValueType.FLOAT)
    ],
    online=True,
    batch_source=PT_source_view_postgres,
    tags={}
)