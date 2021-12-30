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

def run_retrieve_neo4j_table(table_name : String, column_name : list[String] = None) :
    table_query = "MATCH (t:Author) RETURN t.name"
    df_res = None
    if column_name == None : 
        table_query += "t LIMIT 2"
            
    else :
        str_to_append = f"""t.{column_name[0]} """
        table_query += str_to_append
        column_name.pop(0)
        for column in column_name :
            str_to_append = f""", t.{column} """
            table_query += str_to_append
        res_query = driver.run_transaction_query(table_query, run_query=driver.run_query_return_data_key)
        df_res = pd.DataFrame.from_dict(res_query)
    res_query = driver.run_transaction_query(table_query, run_query=driver.run_query_return_data)
    print(res_query)
    res_query = [x["t"] for x in res_query["data"]]
    df_res = pd.DataFrame.from_dict(res_query)
    return df_res



def run_store_data(table_name : String, df_table : pd.DataFrame, if_exist : String) : 
    var_temp = pd.Timestamp.now()
    df_table["event_timestamp"] = var_temp
    df_table["created"] = var_temp
    if if_exist == "replace" : 
        df_table.to_sql(table_name, con, offline_config.db_schema, if_exists="replace", index=False) # rimpiazzo tutta la tabella su postgres
    elif if_exist == "append" : 
        df_table.to_sql(table_name, con, offline_config.db_schema, if_exists="append", index=False) # aggiungo i dati alla tabella su postgres



def run_create_offline_table(table_name : String, df_table : pd.DataFrame, if_exist : String = None):
    var_temp = pd.Timestamp.now()
    df_table["event_timestamp"] = var_temp
    df_table["created"] = var_temp
    if if_exist == "replace" : 
        df_table.to_sql(table_name, con, offline_config.db_schema, if_exists="replace", index=False) # rimpiazzo tutta la tabella su postgres
    else : 
        df_table.to_sql(table_name, con, offline_config.db_schema, index=False) # creo la tabella su postgres


def run_delete_offline_table(table_name : String) :
    con.execute(f"DROP CASCADE TABLE IF EXISTS {table_name}")


