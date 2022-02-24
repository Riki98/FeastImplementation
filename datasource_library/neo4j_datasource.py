import string
from feast_postgres import PostgreSQLOfflineStoreConfig
from typing import Dict, List, Any
from feast import FeatureStore
import driver_neo4j 
import pandas as pd
import numpy as np
import sqlalchemy
import yaml

# Taking the whole configuration for Feast from the yaml file.
# config: Dictionary
with open("./feature_repo/feature_store.yaml", "r") as stream:
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


#mapping type of dataframe
def map_type_python(df_input : pd.DataFrame):
    dict_df = {}
    for i,j in zip(df_input.columns, df_input.dtypes):
        if "object" in str(j):
            if isinstance(df_input[i][len(df_input.index)-1], list) and all(isinstance(e, (int, float)) for e in df_input[i][len(df_input.index)-1]) :
                df_input[i] = df_input[i].astype('np.array')
            elif isinstance(df_input[i][len(df_input.index)-1], str) :
                dict_df.update({i: str})
    return dict_df


# Functions Neo4j

def run_retrieve_neo4j_node(node_name : str) :
    """ 
    This function returns a Pandas DataFrame which represents a group of node in Neo4J DB, retrieved thanks to the driver_neo4j's functions.
    It runs a transaction to get a dictionary with the corrisponding data.

    :node_name: the name of the nodes we want retrieve.
    """
    node_query = f"MATCH (n:{node_name}) RETURN n, id(n) as {node_name}_id_neo4j"
    res_query = driver_neo4j.run_transaction_query(node_query, run_query=driver_neo4j.run_query_return_data)
    temp = res_query["data"]
    res_query = [x["n"] for x in temp]
    id_query = [x[f"{node_name}_id_neo4j"] for x in temp]
    df_res = pd.DataFrame.from_dict(res_query)
    df_id = pd.DataFrame.from_dict(id_query)
    for col in df_res.columns :
        df_res.rename({col : f"{col}_{node_name}"}, axis=1, inplace=True)
    df_res[f"{node_name}_id_neo4j"] = df_id
    
    # CASTA I TIPI AL MOMENTO DEL CARICAMENTO
    """ dict_df_res = map_type_python(df_res)
    print(dict_df_res)
    for col in df_res.columns : 
        if col in dict_df_res :
            df_res.replace({col: dict_df_res[f"{col}"]})
            print({col: dict_df_res[f"{col}"]})

    print(df_res.dtypes) """

    return df_res


def run_retrieve_neo4j_relationship(relationship_name : str) :
    """ 
    This function returns a Pandas DataFrame which represents a group of node in Neo4J DB, retrieved thanks to the driver_neo4j's functions.
    It runs a transaction to get a dictionary with the corrisponding data.

    :node_name: the name of the nodes we want retrieve.
    """
    query_relationship = f"MATCH p=(n)-[r:{relationship_name}]->(m) RETURN r, labels(n), labels(m)"
    rel_res = driver_neo4j.run_transaction_query(query_relationship, run_query=driver_neo4j.run_query_return_value)["values"]
    rel_res.sort(key=(lambda x : len(x[0].keys())))
    relationship_df = pd.DataFrame()
    temp_column_name = set([f"id", "name", "start_node", "end_node", "label(n)", "label(m)"])
    for x in rel_res:
        data_list = {"id" : x[0].id, "name": x[0].type, "start_node":  x[0].start_node.id, "end_node": x[0].end_node.id, "label(n)": x[1][0], "label(m)": x[2][0]}

        properties_dict = zip(x[0].keys(), x[0].values())
        if properties_dict :
            for k, v in properties_dict : 
                temp_column_name.add(k)
                data_list[k] = v

        for i in temp_column_name.difference(set(data_list.keys())) :
            data_list[i] = None
            
        temp = pd.DataFrame([data_list])
        relationship_df = relationship_df.append(temp)

    return relationship_df



#mapping type of dataframe
def map_type_postgres(df_input : pd.DataFrame):
    dict_df = {}
    for i,j in zip(df_input.columns, df_input.dtypes):
        if "object" in str(j):
            if isinstance(df_input[i][len(df_input.index)-1], list) and all(isinstance(e, (int, float)) for e in df_input[i][len(df_input.index)-1]) :
                dict_df.update({i: sqlalchemy.types.ARRAY(sqlalchemy.types.FLOAT())})
            elif isinstance(df_input[i][len(df_input.index)-1], str) :
                dict_df.update({i: sqlalchemy.types.VARCHAR(length=255)})
    return dict_df



# Managing data - offline store

def run_create_offline_node(table_name : str, df_input : pd.DataFrame, if_exists : str = None):
    """ 
    This function uses as input a DataFrame that will make up the Postgres table. 
    The user can decide if replace a table, in case it is already present, or creating it from scratch.
    It also add the columns \"event_timestamp\" and \"created\" for the offline store.

    :table_name: the name of the table the function creates/replaces.
    :df_input: the dataframe the function save as table in Postgres.
    :if_exists: if "None" (as default), the function creates the table instead of replacing it with "replace".
    """

    var_temp = pd.Timestamp.now()
    df_input["event_timestamp"] = var_temp
    df_input["created"] = var_temp

    dict_type = map_type_postgres(df_input)
    df_input.columns= df_input.columns.str.lower()

    if if_exists == "replace" : 
        df_input.to_sql(table_name, con, offline_config.db_schema, if_exists="replace", index=False, chunksize=500, dtype=dict_type)
    elif if_exists == None :
        df_input.to_sql(table_name, con, offline_config.db_schema, index=False, chunksize=500, dtype=dict_type) 
    else : 
        print(f"\"{if_exists}\" not allowed. Did you mean \"replace\"?")

    con.execute(f"ALTER TABLE \"{table_name}\" ADD PRIMARY KEY (\"{table_name.lower()}_id_neo4j\");")
    con.execute(f"CREATE INDEX IF NOT EXISTS \"idx_timestamp\" ON \"{table_name}\"(\"event_timestamp\");")
    con.execute(f"CREATE INDEX IF NOT EXISTS \"idx_created\" ON \"{table_name}\"(\"created\");")
    print(f"{table_name} created")



def run_create_offline_relationship(relationship_name : str, df_input : pd.DataFrame):
    """ 
    This function uses as input a DataFrame that will make up the Postgres table. 
    The user can decide if replace a table, in case it is already present, or creating it from scratch.
    It also add the columns \"event_timestamp\" and \"created\" for the offline store.

    :table_name: the name of the table the function creates/replaces.
    :df_input: the dataframe the function save as table in Postgres.
    :if_exists: if "None" (as default), the function creates the table instead of replacing it with "replace".
    """
    run_drop_offline_table(relationship_name)

    n_start = f'{df_input["label(n)"].iloc[0]}_id_neo4j' 
    n_end = f'{df_input["label(m)"].iloc[0]}_id_neo4j' 

    df_input.rename({"id" : f"{relationship_name}_id_neo4j"}, axis=1, inplace=True)
    df_input.rename({"start_node" : n_start}, axis=1, inplace=True)
    df_input.rename({"end_node" : n_end}, axis=1, inplace=True)
    for col in df_input.columns :
        if col != f"{relationship_name}_id_neo4j" and col != n_start and col != n_end:  
            df_input.rename({col : f"{col}_{relationship_name}"}, axis=1, inplace=True)

    var_temp = pd.Timestamp.now()
    df_input["event_timestamp"] = var_temp
    df_input["created"] = var_temp

    df_input.columns= df_input.columns.str.lower()

    # Spezzo orizzontalmente il dataframe, in modo da avere i pezzi con coppie di nodi dello stesso tipo sullo stesso pezzo.
    # carico il pezzo in una tabella fittizia, inserisco le foreign key e la unisco (UNION) alla tabella principale.
    df_labels = df_input[[f"label(n)_{relationship_name.lower()}", f"label(m)_{relationship_name.lower()}"]]
    df_labels = df_labels.drop_duplicates()
    flag = 0
    for iter, row in df_labels.iterrows() : 
        if(flag == 0) :
            temp = df_input[(df_input[f"label(n)_{relationship_name.lower()}"] == row[f"label(n)_{relationship_name.lower()}"]) & (df_input[f"label(m)_{relationship_name.lower()}"] == row[f"label(m)_{relationship_name.lower()}"])]
            temp.to_sql(f"{relationship_name}_temp", con, offline_config.db_schema, index=False, chunksize=500) 
            con.execute(f"ALTER TABLE \"{relationship_name}_temp\" ADD CONSTRAINT \"costraint_fk_start\" FOREIGN KEY (\"{row[0].lower()}_id_neo4j\") REFERENCES \"{row[0]}\"(\"{row[0].lower()}_id_neo4j\");")
            con.execute(f"ALTER TABLE \"{relationship_name}_temp\" ADD CONSTRAINT \"costraint_fk_end\" FOREIGN KEY (\"{row[1].lower()}_id_neo4j\") REFERENCES \"{row[1]}\"(\"{row[1].lower()}_id_neo4j\");")
            flag+=1
        else :
            temp = df_input[(df_input[f"label(n)_{relationship_name.lower()}"] == row[f"label(n)_{relationship_name.lower()}"]) & (df_input[f"label(m)_{relationship_name.lower()}"] == row[f"label(m)_{relationship_name.lower()}"])]

            temp.to_sql("temp", con, offline_config.db_schema, index=False, chunksize=500) 
            con.execute(f"ALTER TABLE \"temp\" ADD CONSTRAINT \"costraint_fk_start\" FOREIGN KEY (\"{row[0].lower()}_id_neo4j\") REFERENCES \"{row[0]}\"(\"{row[0].lower()}_id_neo4j\");")
            con.execute(f"ALTER TABLE \"temp\" ADD CONSTRAINT \"costraint_fk_end\" FOREIGN KEY (\"{row[1].lower()}_id_neo4j\") REFERENCES \"{row[1]}\"(\"{row[1].lower()}_id_neo4j\");")

            con.execute(f"CREATE TABLE \"temp_table\" AS (SELECT * FROM \"{relationship_name}_temp\" UNION SELECT * FROM \"temp\");")
            con.execute(f"DROP TABLE IF EXISTS \"{relationship_name}_temp\"")
        
    """
    If df_input has only a row of labels, set the correct name deleting the "_temp" part, otherwise change the whole name of temp_table
    """
    if flag == 1 : 
        con.execute(f"ALTER TABLE \"{relationship_name}_temp\" RENAME TO \"{relationship_name}\"")
    else :
        con.execute(f"ALTER TABLE IF EXISTS \"temp_table\" RENAME TO \"{relationship_name}\"")

    con.execute(f"ALTER TABLE \"{relationship_name}\" ADD PRIMARY KEY (\"{relationship_name.lower()}_id_neo4j\");")
    con.execute(f"CREATE INDEX IF NOT EXISTS \"idx_timestamp\" ON \"{relationship_name}\"(\"event_timestamp\");")
    con.execute(f"CREATE INDEX IF NOT EXISTS \"idx_created\" ON \"{relationship_name}\"(\"created\");")
    

def run_drop_offline_table(table_name : str):
    """ 
    Drop a table from Postgres DB, if it exists.

    :table_name: the name of the table to delete.
    """
    con.execute(f"DROP TABLE IF EXISTS \"{table_name}\" CASCADE")


def run_store_data(table_name : str, df_input : pd.DataFrame, if_exists : str = "append"): 
    """ 
    Function used to store the data into the offline store.
    It also add the columns \"event_timestamp\" and \"created\" for the offline store.

    :table_name: the data destination table.
    :df_input: the DataFrame to store.
    """
    dict_type = map_type_postgres(df_input)

    var_temp = pd.Timestamp.now()
    df_input["event_timestamp"] = var_temp
    df_input["created"] = var_temp
    # Checks if the table exists, otherwise raise an error
    try:
        con.execute("select exists(select * from information_schema.tables where table_name=%s)", (table_name)).fetchone()
    except:
        print("The table doesn't exists")
    #fare append e replace

    df_input.to_sql(table_name, con, offline_config.db_schema, if_exists, index=False, dtype=dict_type)
    print(f"Data stored in {table_name}")


def run_retrieve_neo4j_db() :
    """ 
    This function returns a Pandas DataFrame which represents a group of node in Neo4J DB, retrieved thanks to the driver_neo4j's functions.
    It runs a transaction to get a dictionary with the corrisponding data.

    :node_name: the name of the nodes we want retrieve.
    """

    # lista nodi
    query_all_nodes = "MATCH (n) RETURN distinct labels(n)"
    nodes_res = driver_neo4j.run_transaction_query(query_all_nodes, run_query=driver_neo4j.run_query_return_data)
    temp_list = [x["labels(n)"] for x in nodes_res["data"]]
    node_list = []
    # necessario il for altrimenti inseriva "<generator object run_retrieve_neo4j_db.<locals>.<genexpr> at 0x000001C3DA83CA50>"
    for x in temp_list:
        node_list.append(x[0])

    # caricamento nodi
    for node in node_list:
        node_table = run_retrieve_neo4j_node(node)
        run_create_offline_node(node, node_table)


    # lista relazioni
    query_list_relationship = "MATCH p=()-[r]->() RETURN r"
    list_res = driver_neo4j.run_transaction_query(query_list_relationship, run_query=driver_neo4j.run_query_return_data)
    relationship_list = set([x["r"][1] for x in list_res["data"]])
    
    # caricamento relazioni
    for rel in relationship_list :
        relationship_df = run_retrieve_neo4j_relationship(rel)
        run_create_offline_relationship(rel, relationship_df)


# Retrieve from offline store

def get_online_feature(feature_list : List[str], entity_rows : List[Dict[str, Any]]):
    
    store = FeatureStore(repo_path="./feature_repo")
    
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
