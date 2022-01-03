import numpy as np
import pandas as pd
import neo4j_datasource

###################################################################################


df = neo4j_datasource.run_retrieve_neo4j_table("Author")
print(df)
neo4j_datasource.run_store_data("test", df, "append")
#neo4j_datasource.run_drop_offline_table("Prova")