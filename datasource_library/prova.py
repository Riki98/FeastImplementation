import neo4j_datasource

""" import time

start = time.time()
stop = time. time()
print("The time of the run:", stop - start) """


# Inserisco i dati da neo4j a postgres

#neo4j_datasource.run_drop_offline_table("Author")

#neo4j_datasource.run_retrieve_neo4j_db()

#neo4j_datasource.run_create_offline_table("Author", df_auth, "replace")
#neo4j_datasource.run_store_data("Author", df_auth)

#df_auth = neo4j_datasource.run_retrieve_neo4j_node("Author")
#neo4j_datasource.run_create_offline_node("Author", df_auth)
#neo4j_datasource.run_create_offline_table("Paper", df_auth, "replace")

#neo4j_datasource.run_retrieve_neo4j_node("Author")

#neo4j_datasource.rinomina()

######### ONLINE

"""entity_rows=[
        {"Author_id": "546"},
        {"Author_id": "547"}
    ]

print("\n\n")
print("ONLINE REDIS RETRIEVAL")
df_offline = neo4j_datasource.get_online_feature(feature_list=feature_auth, entity_rows=entity_rows)
print("\n\n") """
