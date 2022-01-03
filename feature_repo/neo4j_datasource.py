import pandas as pd
from sqlalchemy.sql.sqltypes import String
import yaml
import sqlalchemy
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
def run_retrieve_neo4j_table(table_name : String, column_name : list[String] = None) :
    table_query = f"MATCH (t:{table_name}) RETURN "
    df_res = None
    if column_name == None : 
        table_query += "t LIMIT 2"
    res_query = driver.run_transaction_query(table_query, run_query=driver.run_query_return_data)
    print(res_query)
    res_query = [x["t"] for x in res_query["data"]]
    df_res = pd.DataFrame.from_dict(res_query)
    return df_res


# funziona
def run_create_offline_table(table_name : String, df_table : pd.DataFrame, if_exist : String = None):
    var_temp = pd.Timestamp.now()
    df_table["event_timestamp"] = var_temp
    df_table["created"] = var_temp
    if if_exist == "replace" : 
        df_table.to_sql(table_name, con, offline_config.db_schema, if_exists="replace") # rimpiazzo tutta la tabella su postgres
    else :
        df_table.to_sql(table_name, con, offline_config.db_schema) # creo la tabella su postgres

#funziona, anche se maiuscole e minuscole nel nome della tabella non fanno differenza
def run_drop_offline_table(table_name : String) :
    print(table_name)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")

# se append non è specificato, creo una nuova tabello su cui fare lo store dei dati
def run_store_data(table_name : String, df_table : pd.DataFrame, if_exist : String = None) : 
    var_temp = pd.Timestamp.now()
    df_table["event_timestamp"] = var_temp
    df_table["created"] = var_temp
    if if_exist == "append" : 
        df_table.to_sql(table_name, con, offline_config.db_schema, if_exists="append") # aggiungo i dati alla tabella su postgres
    else : 
        df_table.to_sql(table_name, con, offline_config.db_schema) # creo una tabella su postgres