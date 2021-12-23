from io import StringIO
import pandas as pd
import numpy as np
from sqlalchemy.sql.sqltypes import FLOAT, DECIMAL, TEXT, TIMESTAMP
import yaml
import sqlalchemy
from feast_postgres import PostgreSQLOfflineStoreConfig
import neo4j_config as neo4j_config
from neo4j import GraphDatabase
from sqlalchemy.types import ARRAY



############################################################ QUI IMPORTA GLI EMBEDDING DA NEO4J ##################################################################
auth_id, auth_emb = neo4j_config.get_authors()
timestamp_auth = [pd.Timestamp.now() for emb in auth_emb]
df_auth = pd.DataFrame(index=auth_id, columns=["Authors_embedding"])
df_auth["Authors_embedding"] = [np.array(emb) for emb in auth_emb]
df_auth["event_timestamp"] = timestamp_auth
df_auth["created"] = timestamp_auth

############################################################# CONFIGURAZIUONE OFFLINE ############################################################################
with open("feature_store.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


offline_config = config["offline_store"]
del offline_config["type"]
offline_config = PostgreSQLOfflineStoreConfig(**offline_config)

def get_sqlalchemy_engine(config: PostgreSQLOfflineStoreConfig):
    url = f"postgresql+psycopg2://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}" 
    print(url)
    print(config.db_schema)
    return sqlalchemy.create_engine(url, client_encoding='utf8', connect_args={'options': '-c search_path={}'.format(config.db_schema)}) 
con = get_sqlalchemy_engine(offline_config)


############################################################ CREAZIONE TABELLE SU POSTGRES ##################################################################

table = "authors"
con.execute("DROP TABLE IF EXISTS " + table)
df_auth.to_sql(table, con, dtype={"Authors_embedding": ARRAY(FLOAT), "event_timestamp": TIMESTAMP, "created": TIMESTAMP})
res = con.execute("SELECT * FROM authors").fetchone()
print(type(res["Authors_embedding"]))