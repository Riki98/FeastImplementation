from os import path
from google.protobuf.duration_pb2 import Duration
from feast import Entity, Feature, FeatureView, ValueType
from feast_postgres import PostgreSQLSource

################################################## AUTHOR  

#creo le view da postgres
authors_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"Author\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)

# Define an entity for the auth
index_entity_auth = Entity(name="author_id_neo4j", value_type=ValueType.INT64, description="author id neo4j")


# postgres e redis usano le stesse identiche feature view
author_embedding_view_postgres = FeatureView(
    name="authors_view",
    entities=["author_id_neo4j"],
    ttl=Duration(seconds=8000 * 1),
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

#creo le view da postgres
papers_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"Paper\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)

# Define an entity for the auth
index_entity_paper = Entity(name="paper_id_neo4j", value_type=ValueType.INT64, description="paper id neo4j")


# postgres e redis usano le stesse identiche feature view
paper_embedding_view_postgres = FeatureView(
    name="papers_view",
    entities=["paper_id_neo4j"],
    ttl=Duration(seconds=8000 * 1),
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

#creo le view da postgres
conference_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"Conference\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)

# Define an entity for the conference
index_entity_conf = Entity(name="Conference_id_neo4j", value_type=ValueType.INT64, description="conference id neo4j")


# postgres e redis usano le stesse identiche feature view
conference_embedding_view_postgres = FeatureView(
    name="conference_view",
    entities=["Conference_id_neo4j"],
    ttl=Duration(seconds=8000 * 1),
    features=[
        Feature(name="name_Conference", dtype=ValueType.STRING),
        Feature(name="label_Conference", dtype=ValueType.FLOAT),
        Feature(name="Conference_id_neo4j", dtype=ValueType.STRING),
        Feature(name="graphsage_embedding_Conference", dtype=ValueType.FLOAT_LIST),
        Feature(name="fastrp_embedding_Conference", dtype=ValueType.FLOAT_LIST)
    ],
    online=True,
    batch_source=conference_source_view_postgres,
    tags={}
)


################################################## TERM

#creo le view da postgres
term_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"Term\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)

# Define an entity for the auth
index_entity_term = Entity(name="Term_id_neo4j", value_type=ValueType.INT64, description="term id neo4j")


# postgres e redis usano le stesse identiche feature view
term_embedding_view_postgres = FeatureView(
    name="term_view",
    entities=["Term_id_neo4j"],
    ttl=Duration(seconds=8000 * 1),
    features=[
        Feature(name="name_Term", dtype=ValueType.STRING),
        Feature(name="label_Term", dtype=ValueType.FLOAT),
        Feature(name="Term_id_neo4j", dtype=ValueType.STRING),
        Feature(name="graphsage_embedding_Term", dtype=ValueType.FLOAT_LIST),
        Feature(name="fastrp_embedding_Term", dtype=ValueType.FLOAT_LIST)
    ],
    online=True,
    batch_source=term_source_view_postgres,
    tags={}
)


################################################## PA

#creo le view da postgres
PA_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"PA\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)


# postgres e redis usano le stesse identiche feature view
PA_embedding_view_postgres = FeatureView(
    name="pa_view",
    entities=["paper_id_neo4j", "author_id_neo4j"],
    ttl=Duration(seconds=8000 * 1),
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

#creo le view da postgres
PC_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"PC\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)


# postgres e redis usano le stesse identiche feature view
PC_embedding_view_postgres = FeatureView(
    name="PC_view",
    entities=["Paper_id_neo4j", "Conference_id_neo4j"],
    ttl=Duration(seconds=8000 * 1),
    features=[
        Feature(name="PC_id_neo4j", dtype=ValueType.INT64),
        Feature(name="name_PC", dtype=ValueType.STRING),
        Feature(name="Paper_id_neo4j_PC", dtype=ValueType.INT64),
        Feature(name="Conference_id_neo4j_PC", dtype=ValueType.INT64),
        Feature(name="label(n)_PC", dtype=ValueType.STRING),
        Feature(name="label(m)_PC", dtype=ValueType.STRING),
        Feature(name="label_n_PC", dtype=ValueType.FLOAT)
    ],
    online=True,
    batch_source=PC_source_view_postgres,
    tags={}
)


################################################## PT

#creo le view da postgres
PT_source_view_postgres = PostgreSQLSource(
    query="SELECT * FROM \"PT\"", 
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)


# postgres e redis usano le stesse identiche feature view
PT_embedding_view_postgres = FeatureView(
    name="PT_view",
    entities=["Paper_id_neo4j", "Term_id_neo4j"],
    ttl=Duration(seconds=8000 * 1),
    features=[
        Feature(name="PT_id", dtype=ValueType.INT64),
        Feature(name="name_PT", dtype=ValueType.STRING),
        Feature(name="Paper_id_neo4j_PT", dtype=ValueType.INT64),
        Feature(name="Term_id_neo4j_PT", dtype=ValueType.INT64),
        Feature(name="label(n)_PT", dtype=ValueType.STRING),
        Feature(name="label(m)_PT", dtype=ValueType.STRING),
        Feature(name="label_n_PT", dtype=ValueType.FLOAT)
    ],
    online=True,
    batch_source=PT_source_view_postgres,
    tags={}
)