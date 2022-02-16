import dgl
import numpy as np
import torch
import neo4j_datasource as nd
from sqlalchemy.sql.sqltypes import String
from typing import Dict, List, Any
from feast import FeatureStore

# le feature da prendere:
feature_pa = [
    "papers_view:name_paper",
    "papers_view:label_paper",
    "papers_view:paper_id_neo4j",
    "pa_view:label_n_pa",
    "authors_view:author_id_neo4j",
    "authors_view:name_author",
    "authors_view:label_author"
]


store = FeatureStore(repo_path="./feature_repo")

query = f"SELECT \"paper_id_neo4j\", \"author_id_neo4j\", \"event_timestamp\" FROM \"PA\" LIMIT 10"

pa_df = store.get_historical_features(
    entity_df=query,
    features=feature_pa,
    full_feature_names=True
).to_arrow()

print(pa_df)



#creo dei tensori pytorch da postgres

# creo numpy array dalle colonne dei foreign key
array_n = np.array(pa_df["paper_id_neo4j"])
array_m = np.array(pa_df["author_id_neo4j"])


# creo il tensore
tensor_n = torch.from_numpy(array_n)
tensor_m = torch.from_numpy(array_m)

#creo il grafo con dgl, inderendo come 'data' i tensori creati
g = dgl.graph((tensor_n, tensor_m), num_nodes=(len(array_n) + len(array_m)))

#vedo che c'è
print(g)

# A DGL graph can store node features and edge features in two dictionary-like attributes called 'ndata' and 'edata'
print("pa_df[\"name_Author\"]")
print(pa_df["name_Author"].head())
g.ndata["name_Author"] = torch.from_numpy(np.array(pa_df["name_Author"]))