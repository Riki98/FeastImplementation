import neo4j_datasource

import time

start = time.time()


# Inserisco i dati da neo4j a postgres

#neo4j_datasource.run_drop_offline_table("Author")
df_auth = neo4j_datasource.run_retrieve_neo4j_db()
#neo4j_datasource.run_create_offline_table("Author", df_auth, "replace")
#neo4j_datasource.run_store_data("Author", df_auth)

#df_auth = neo4j_datasource.run_retrieve_neo4j_node("Paper")
#neo4j_datasource.run_create_offline_table("Paper", df_auth, "replace")


stop = time. time()
print("The time of the run:", stop - start)


######## Def
feature_auth = [
    "authors_view:graphsage_embedding",
    "authors_view:fastrp_embedding"
]

######## OFFLINE

""" print("\n\n")
print("OFFLINE POSTGRES RETRIEVAL")
df_offline = neo4j_datasource.get_offline_feature(feature_list=feature_auth, table="Author")
print(df_offline)
print("\n\n")

######### ONLINE

entity_rows=[
        {"index_auth": "A76"},
        {"index_auth": "A124"}
    ]

print("\n\n")
print("ONLINE REDIS RETRIEVAL")
df_offline = neo4j_datasource.get_online_feature(feature_list=feature_auth, entity_rows=entity_rows)
print("\n\n") """