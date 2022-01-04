import neo4j_datasource


df = neo4j_datasource.run_retrieve_neo4j_table("Author")
#neo4j_datasource.run_create_offline_table("test", df, "replace") #funziona
neo4j_datasource.run_store_data("test", df)
#neo4j_datasource.run_drop_offline_table("test")