import yaml
import sqlalchemy
import pandas as pd
import driver_neo4j as driver
from sqlalchemy.sql.sqltypes import String
from feast_postgres import PostgreSQLOfflineStoreConfig


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
    node_query = f"MATCH (n:{node_name}) RETURN n"
    res_query = driver.run_transaction_query(node_query, run_query=driver.run_query_return_data)
    res_query = [x["n"] for x in res_query["data"]]
    df_res = pd.DataFrame.from_dict(res_query)
    return df_res


# Managing data

def run_create_offline_table(table_name : String, df_input : pd.DataFrame, if_exist : String = None):
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
        df_input.to_sql(table_name, con, offline_config.db_schema, if_exists="replace", index=False) 
    elif if_exist == None :
        df_input.to_sql(table_name, con, offline_config.db_schema, index=False) 
    else : 
        print(f"\"{if_exist}\" not allowed. Did you mean \"replace\"?")
    print(f"{table_name} created")


def run_drop_offline_table(table_name : String):
    """ 
    Function used to drop a table from Postgres DB, if it exists.

    :table_name: the name of the table to delete.
    """
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    print(f"{table_name} dropped")


def run_store_data(table_name : String, df_input : pd.DataFrame): 
    """ 
    run_store_data: the function used to store the data into the offline store.
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

    df_input.to_sql(table_name, con, offline_config.db_schema, if_exists="append", index=False)
    print(f"Data stored in {table_name}")