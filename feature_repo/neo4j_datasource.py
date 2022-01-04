import pandas as pd
from sqlalchemy.sql.schema import Table
from sqlalchemy.sql.sqltypes import String
import yaml
import json
import sqlalchemy
from sqlalchemy.orm import declarative_base
from sqlalchemy import update
from feast_postgres import PostgreSQLOfflineStoreConfig
import driver_neo4j as driver

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
    print(config.db_schema)
    return sqlalchemy.create_engine(url, client_encoding='utf8', connect_args={'options': '-c search_path={}'.format(config.db_schema)}) 
con = get_sqlalchemy_engine(offline_config)


############################################################ FUNZIONI NEO4J ##################################################################

# funziona
def run_retrieve_neo4j_table(table_name : String) :
    table_query = f"MATCH (t:{table_name}) RETURN t LIMIT 4"
    res_query = driver.run_transaction_query(table_query, run_query=driver.run_query_return_data)
    res_query = [x["t"] for x in res_query["data"]]
    df_res = pd.DataFrame.from_dict(res_query)
    return df_res


# funziona
def run_create_offline_table(table_name : String, df_input : pd.DataFrame, if_exist : String = None):
    var_temp = pd.Timestamp.now()
    df_input["event_timestamp"] = var_temp
    df_input["created"] = var_temp
    if if_exist == "replace" : 
        df_input.to_sql(table_name, con, offline_config.db_schema, if_exists="replace", index=False) # rimpiazzo tutta la tabella su postgres
    else :
        df_input.to_sql(table_name, con, offline_config.db_schema, index=False) # creo la tabella su postgres



# funziona, anche se maiuscole e minuscole nel nome della tabella non fanno differenza
def run_drop_offline_table(table_name : String) :
    print(table_name)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")



# da provare
def run_store_data(table_name : String, df_input : pd.DataFrame) : 
    var_temp = pd.Timestamp.now()
    df_input["event_timestamp"] = var_temp
    df_input["created"] = var_temp
    #controllo se esiste la tabella, in caso restituisco sollevo un errore
    try:
        con.execute("select exists(select * from information_schema.tables where table_name=%s)", (table_name)).fetchone()
    except:
        print("The table doesn't exists")
    #casi if_exist
    # aggiungo i dati alla tabella su postgres
    df_input.to_sql(table_name, con, offline_config.db_schema, if_exists="append", index=False)