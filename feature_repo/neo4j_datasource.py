from feast_postgres import PostgreSQLOfflineStoreConfig
from sqlalchemy.sql.sqltypes import String
from typing import Dict, List, Any
from feast import FeatureStore
from tempfile import tempdir
import driver_neo4j as driver
from neo4j import graph
import pandas as pd
import sqlalchemy
import time
import yaml


# Taking the whole configuration for Feast from the yaml file.
# config: Dictionary
with open("feature_store.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
offline_config = config["offline_store"]
del offline_config["type"]
offline_config = PostgreSQLOfflineStoreConfig(**offline_config)

# Postgres connection setup

def get_sqlalchemy_engine(config: PostgreSQLOfflineStoreConfig):
    """ 
    get_sqlalchemy_engine is used to set up the PostgreSQL DBMS astraction. Returns the interface with which we will make the queries.
    :config: PostgreSQLOfflineStoreConfig class
     """
    url = f"postgresql+psycopg2://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}" 
    return sqlalchemy.create_engine(url, client_encoding='utf8', connect_args={'options': '-c search_path={}'.format(config.db_schema)}) 
con = get_sqlalchemy_engine(offline_config)
con.execute(f"CREATE SCHEMA IF NOT EXISTS {offline_config.db_schema}") # create the schema in Postgres DB if not present


# Functions Neo4j

def run_retrieve_neo4j_node(node_name : String) :
    """ 
    This function returns a Pandas DataFrame which represents a group of node in Neo4J DB, retrieved thanks to the driver_neo4j's functions.
    It runs a transaction to get a dictionary with the corrisponding data.

    :node_name: the name of the nodes we want retrieve.
    """
    node_query = f"MATCH (n:{node_name}) RETURN n, id(n)"
    res_query = driver.run_transaction_query(node_query, run_query=driver.run_query_return_data)
    res_query = [x["n"] for x in res_query["data"]]
    df_res = pd.DataFrame.from_dict(res_query)
    return df_res

# Managing data - offline store

def run_create_offline_node(table_name : String, df_input : pd.DataFrame, if_exist : String = None):
    """ 
    This function uses as input a DataFrame that will make up the Postgres table. 
    The user can decide if replace a table, in case it is already present, or creating it from scratch.
    It also add the columns \"event_timestamp\" and \"created\" for the offline store.

    :table_name: the name of the table the function creates/replaces.
    :df_input: the dataframe the function save as table in Postgres.
    :if_exist: if "None" (as default), the function creates the table instead of replacing it with "replace".
    """
    var_temp = pd.Timestamp.now()
    df_input["event_timestamp"] = var_temp
    df_input["created"] = var_temp
    if if_exist == "replace" : 
        df_input.to_sql(table_name, con, offline_config.db_schema, if_exists="replace", index=False, chunksize=500) 
    elif if_exist == None :
        df_input.to_sql(table_name, con, offline_config.db_schema, index=False, chunksize=500) 
    else : 
        print(f"\"{if_exist}\" not allowed. Did you mean \"replace\"?")

    con.execute(f"ALTER TABLE {table_name} ADD PRIMARY KEY (\"id(n)\");")
    con.execute(f"CREATE INDEX IF NOT EXIST idx_timestamp ON \"{table_name}\"(\"event_timestamp\");")
    con.execute(f"CREATE INDEX IF NOT EXIST idx_created ON \"{table_name}\"(\"created\");")
    print(f"{table_name} created")

def run_create_offline_relationship(relationship_name : String, df_input : pd.DataFrame, if_exist : String = None):
    """ 
    This function uses as input a DataFrame that will make up the Postgres table. 
    The user can decide if replace a table, in case it is already present, or creating it from scratch.
    It also add the columns \"event_timestamp\" and \"created\" for the offline store.

    :table_name: the name of the table the function creates/replaces.
    :df_input: the dataframe the function save as table in Postgres.
    :if_exist: if "None" (as default), the function creates the table instead of replacing it with "replace".
    """
    var_temp = pd.Timestamp.now()
    df_input["event_timestamp"] = var_temp
    df_input["created"] = var_temp
    #n_list = set([x[0] for x in df_input["labels(n)"]])
    #m_list = set([x[0] for x in df_input["labels(m)"]])

    if if_exist == "replace" : 
        df_input.to_sql(relationship_name, con, offline_config.db_schema, if_exists="replace", index=False, chunksize=500) 
    elif if_exist == None :
        df_input.to_sql(relationship_name, con, offline_config.db_schema, index=False, chunksize=500) 
    else : 
        print(f"\"{if_exist}\" not allowed. Did you mean \"replace\"?")


    con.execute(f"ALTER TABLE {relationship_name} ADD CONSTRAINT costraint_fk_n FOREIGN KEY (id(n)) REFERENCES {start_n}(id(n));")
    con.execute(f"ALTER TABLE {relationship_name} ADD CONSTRAINT costraint_fk_m FOREIGN KEY (id(m)) REFERENCES {end_m}(id(m));")
    con.execute(f"CREATE INDEX IF NOT EXIST idx_timestamp ON \"{relationship_name}\"(\"event_timestamp\");")
    con.execute(f"CREATE INDEX IF NOT EXIST idx_created ON \"{relationship_name}\"(\"created\");")
    print(f"{relationship_name} created")


def run_drop_offline_table(table_name : String):
    """ 
    Function used to drop a table from Postgres DB, if it exists.

    :table_name: the name of the table to delete.
    """
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    print(f"{table_name} dropped")


def run_store_data(table_name : String, df_input : pd.DataFrame): 
    """ 
    Function used to store the data into the offline store.
    It also add the columns \"event_timestamp\" and \"created\" for the offline store.

    :table_name: the data destination table.
    :df_input: the DataFrame to store.
    """
    var_temp = pd.Timestamp.now()
    df_input["event_timestamp"] = var_temp
    df_input["created"] = var_temp
    # Checks if the table exists, otherwise raise an error
    try:
        con.execute("select exists(select * from information_schema.tables where table_name=%s)", (table_name)).fetchone()
    except:
        print("The table doesn't exists")
    #fare append e replace
    df_input.to_sql(table_name, con, offline_config.db_schema, if_exists="append", index=False)
    print(f"Data stored in {table_name}")


def run_retrieve_neo4j_db() :
    """ 
    This function returns a Pandas DataFrame which represents a group of node in Neo4J DB, retrieved thanks to the driver_neo4j's functions.
    It runs a transaction to get a dictionary with the corrisponding data.

    :node_name: the name of the nodes we want retrieve.

    start = time.time()
    stop = time. time()
    print("The time of the run:", stop - start)
    """

    """ # lista nodi
    query_all_nodes = "MATCH (n) RETURN distinct labels(n)"
    nodes_res = driver.run_transaction_query(query_all_nodes, run_query=driver.run_query_return_data)
    temp_list = [x["labels(n)"] for x in nodes_res["data"]]
    node_list = []
    # necessario il for altrimenti inseriva "<generator object run_retrieve_neo4j_db.<locals>.<genexpr> at 0x000001C3DA83CA50>"
    for x in temp_list:
        node_list.append(x[0])
    print(node_list)

    # caricamento nodi
    for node in node_list:
        node_table = run_retrieve_neo4j_node(node)
        run_create_offline_node(node, node_table) """


    # lista relazioni
    query_list_relationship = "MATCH p=()-[r]->() RETURN r"
    list_res = driver.run_transaction_query(query_list_relationship, run_query=driver.run_query_return_data)
    relationship_list = set([x["r"][1] for x in list_res["data"]])
    print(relationship_list)
    relationship_list 

    # caricamento relazioni
    for rel in relationship_list :
        query_relationship = f"MATCH p=(n)-[r:{rel}]->(m) RETURN r, labels(n), labels(m) limit 2"
        rel_res = driver.run_transaction_query(query_relationship, run_query=driver.run_query_return_value)["values"]
        final = pd.DataFrame()
        for x in rel_res:
            temp = pd.DataFrame()
            print(x)
            temp_list = []
            temp_list.append(x[0].id)
            temp_list.append(x[0].type)
            temp_list.append(x[0].start_node.id)
            temp_list.append(x[0].end_node.id)
            for k, v in zip(x[0].keys(), x[0].values()) : 
                temp_list(v)
            
            print(temp_list)

            temp["id"] = x[0].id
            temp["name"] = x[0].type
            temp["start_node"] = x[0].start_node.id
            temp["end_node"] = x[0].end_node.id
            for k, v in zip(x[0].keys(), x[0].values()) : 
                temp[("" + k)] = v
            temp["label(n)"] = x[1]
            temp["label(m)"] = x[2]
            print(temp.columns)
            final = final.append(temp)
        print("Final:")
        print(final)
    
    return

    


# Retrieve from offline store

def get_offline_feature(feature_list : List[String], table : String): 

    store = FeatureStore(repo_path=".")
    
    # query from feature_list and table_list
    query = f"select \"id\", \"event_timestamp\", \"created\" from \"{table}\""
    print(query)
    print("\n\n")

    training_df = store.get_historical_features(
        entity_df=query,
        features=feature_list,
    ).to_df()

    print("----- Feature schema -----\n")
    print(training_df.info())

    print()
    print("----- Example features -----\n")
    print(training_df.head())

    return training_df


# Retrieve from offline store

def get_online_feature(feature_list : List[String], entity_rows : List[Dict[str, Any]]):
    
    store = FeatureStore(repo_path="./")
    
    print(feature_list)

    model_df = store.get_online_features(
        features=feature_list,
        entity_rows=entity_rows
    ).to_df()

    print("----- Feature schema -----\n")
    print(model_df.info())

    print()
    print("----- Example features -----\n")
    print(model_df.head())

    return model_df