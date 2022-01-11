import neo4j
import config
from neo4j import GraphDatabase
from typing import Dict, List


driver = None
default_db = config.NEO4J_DB


# initializes the driver, if it has not been initialized, and returns a session
def session(database=default_db, bookmarks=(), default_access_mode=neo4j.WRITE_ACCESS, fetch_size=1000):
    global driver
    if driver is None:
        driver = init_neo4j_connector()
    return driver.session(database=database, bookmarks=bookmarks, default_access_mode=default_access_mode,
                          fetch_size=fetch_size)


# initialize the neo4j driver with environment variables
# (defined in docker-compose) or in the configuration file config.py
def init_neo4j_connector():
    NEO4J_USER = config.NEO4J_USER 
    NEO4J_PASSWORD = config.NEO4J_PASSWORD 
    auth = (NEO4J_USER, NEO4J_PASSWORD)

    NEO4J_URI = config.NEO4J_URI
    # probably in the future, a resolver should be defined for the driver constructor
    # ex: GraphDatabase.driver(NEO4J_URI, auth=auth, resolver=custom_resolver)
    return GraphDatabase.driver(NEO4J_URI, auth=auth)



def run_query_return_data_key(tx, r_query, params):
    res = tx.run(r_query, params=params)
    return {'keys': res.keys(), 'data': res.data(), 'dataBase': res.consume().database}

def run_query_return_data(tx, r_query, params):
    res = tx.run(r_query, params=params)
    return {'data': res.data(), 'dataBase': res.consume().database}

def run_query_return_value(tx, r_query, params):
    res = tx.run(r_query, params=params)
    return {'values': res.values(), 'dataBase': res.consume().database}

def run_query_return_single(tx, r_query, params):
    res = tx.run(r_query, params=params)
    return {'single': res.single(), 'dataBase': res.consume().database}


def run_query_return_graph(tx, r_query, params):
    res = tx.run(r_query, params=params)
    return {'graph': res.graph(), 'dataBase': res.consume().database}

def run_query_return_value_keys(tx, r_query, params):
    res = tx.run(r_query, params=params)
    return {'values': res.values(), 'keys': res.keys(), 'dataBase': res.consume().database}


def run_transaction_query(cypher_query, params={}, db=default_db, run_query=run_query_return_data, write=False):
    db = (db if not db == "" and not db == " " else default_db)
    print(f"LOG - INFO [neoj4_utils - query]: running query on {db if db else 'default'} DB instance: {cypher_query}")
    with session(database=db) as ssn:
        if not write:
            result = ssn.read_transaction(run_query, cypher_query, params)
        else:
            result = ssn.write_transaction(run_query, cypher_query, params)
        return result
