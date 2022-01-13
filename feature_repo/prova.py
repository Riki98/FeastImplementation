import neo4j_datasource

######## Def
feature_auth = [
    "authors_view:graphsage_embedding",
    "authors_view:fastrp_embedding"
]

######## OFFLINE

print("\n\n")
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
print("\n\n")