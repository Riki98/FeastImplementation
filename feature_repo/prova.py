import numpy as np
import pandas as pd
import neo4j_datasource

###################################################################################

#df = neo4j_datasource.run_retrieve_postgres_table("test")
df = neo4j_datasource.run_retrieve_neo4j_table("Author")
print(df["id"][0])
print("\n\n")
print(df["id"][1])
#neo4j_datasource.run_store_data("test", df, "replace")
#neo4j_datasource.run_drop_offline_table("Prova")