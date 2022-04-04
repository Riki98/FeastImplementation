import os
import dgl
import torch
import time
import dgl.data
import numpy as np
import driver_neo4j
import pandas as pd
from datetime import datetime
from feast import FeatureStore
from rgcn_gnn import HeteroRGCN
import torch.nn.functional as F
from driver_neo4j import run_query_return_data
from sklearn.model_selection import train_test_split


def dict_col(df_input : pd.DataFrame) : 
    dict_col = {}
    for col in df_input.columns :
        if "paper_view__paper_id_neo4j" in col :
            dict_col.update({col: "start_id"})
        elif "author_view__author_id_neo4j" in col or "conference_view__conference_id_neo4j" in col or "term_view__term_id_neo4j" in col :
            dict_col.update({col: "end_id"})
        else :
            dict_col.update({col: col})
    return dict_col



start = time.time()

################################################################# RECUPERO DATAFRAME RELAZIONI ####################################################################
pa_df = pd.read_pickle("pa_df.pkl")
pc_df = pd.read_pickle("pc_df.pkl")
pt_df = pd.read_pickle("pt_df.pkl")

store = FeatureStore(repo_path="./feature_repo")

#################################################################################################################################################################

os.system(f"cd .\\feature_repo\\ && feast teardown")
os.system(f"cd .\\feature_repo\\ && feast apply")

#################################################################################################################################################################

relationships = {}
node_labels_map_neo4j_dgl = {}
node_labels_counter = {}

def map_id(df_input : pd.DataFrame, triplet):
    
    src_label = triplet[0]
    dst_label = triplet[2]
    
    if node_labels_map_neo4j_dgl.get(src_label) is None:
        node_labels_map_neo4j_dgl[src_label] = {}
        node_labels_counter[src_label] = 0
            
    if node_labels_map_neo4j_dgl.get(dst_label) is None:
        node_labels_map_neo4j_dgl[dst_label] = {}
        node_labels_counter[dst_label] = 0

    if relationships.get(triplet) is None:
        relationships[triplet] = []
        relationships[(triplet[2], triplet[1]+'_reversed', triplet[0])] = []
    
    
    for index, element in df_input.iterrows():
        start_id = node_labels_map_neo4j_dgl[src_label].get(element['start_id'])
        if start_id is None:
            node_labels_map_neo4j_dgl[src_label][element['start_id']] = node_labels_counter[src_label]
            start_id = node_labels_counter[src_label]
            node_labels_counter[src_label] += 1

        end_id = node_labels_map_neo4j_dgl[dst_label].get(element['end_id'])
        if end_id is None:
            node_labels_map_neo4j_dgl[dst_label][element['end_id']] = node_labels_counter[dst_label]
            end_id = node_labels_counter[dst_label]
            node_labels_counter[dst_label] += 1

        relationships[triplet].append(
                (start_id, end_id)
            )
        
        relationships[(triplet[2],triplet[1]+'_reversed',triplet[0])].append(
            (end_id, start_id)
        )

map_id(pa_df, ('Paper','PA','Author'))
map_id(pc_df, ('Paper','PC','Conference'))
map_id(pt_df, ('Paper','PT','Term'))

node_labels_map_dgl_neo4j =  {k: {v1: k1 for k1, v1 in v.items()} for k,v in node_labels_map_neo4j_dgl.items()}
graph = dgl.heterograph(relationships)
print(graph)

# Assegno le feature ai nodi
# embeddings
# PROBLEMA: prendo gli embeddings dalla relazione, non va bene

for label in graph.ntypes:
    
    neo4j_label_ids_number = [ node_labels_map_dgl_neo4j[label][x] for x in graph.nodes(label).numpy()]

    entity_df=f"select distinct on (\"{label.lower()+'_id_neo4j'}\") \"{label.lower()+'_id_neo4j'}\", \"created\" as event_timestamp FROM \"{label}\" WHERE \"created\" > now() - INTERVAL '30 days' order by \"{label.lower()+'_id_neo4j'}\", event_timestamp desc"
    label_embedding_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            f"{label.lower()}_view:graphsage_embedding_{label.lower()}"
        ]
    ).to_df()

    label_embedding_df = label_embedding_df.set_index(label.lower()+'_id_neo4j').reindex(neo4j_label_ids_number)
    # label_embedding_df.to_pickle(f"{label.lower()}_graphsage.pkl")
    graph.nodes[label].data['embedding'] = torch.FloatTensor(np.stack(label_embedding_df[f'graphsage_embedding_{label.lower()}'].values, axis=0))


query = f"MATCH(p:Paper) RETURN Id(p) as id, p.label as label"
train_table = driver_neo4j.run_transaction_query(query, run_query=run_query_return_data) # lista di dizionari

trainset = list(zip((x['id'] for x in train_table["data"]),(x['label'] for x in train_table["data"])))

trainset_ids = [] 
trainset_labels =[]
for k,v in trainset:
    trainset_ids.append(k)
    trainset_labels.append(v)

train_neo4j, test_neo4j, train_labels, test_labels = train_test_split(trainset_ids, trainset_labels, train_size=0.6, test_size=0.4, random_state=0, stratify=trainset_labels)
train_neo4j, val_neo4j, train_labels, val_labels = train_test_split(train_neo4j, train_labels, train_size=0.80, test_size=0.20, random_state=0, stratify=train_labels)

train = [node_labels_map_neo4j_dgl['Paper'][x] for x in train_neo4j]
val = [node_labels_map_neo4j_dgl['Paper'][x] for x in val_neo4j]
test =  [node_labels_map_neo4j_dgl['Paper'][x] for x in test_neo4j]

train_labels = torch.LongTensor(train_labels)
val_labels = torch.LongTensor(val_labels)
test_labels = torch.LongTensor(test_labels)

################ ALLENO IL MODELLO CON graphsage

model = HeteroRGCN(graph, 256, 64, 4)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

best_val_acc_graphsage = 0
best_modle_dict_graphsage = None

for epoch in range(100):
    logits = model(graph)
    # The loss is computed only for labeled nodes.
    loss = F.cross_entropy(logits[train], train_labels)

    pred = logits.argmax(1)
    train_acc = (pred[train] == train_labels).float().mean()
    val_acc = (pred[val] == val_labels).float().mean()

    if best_val_acc_graphsage < val_acc:
        best_val_acc_graphsage = val_acc
        best_modle_dict_graphsage = model.state_dict()

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 5 == 0:
        print('Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f)' % (
            loss.item(),
            train_acc.item(),
            val_acc.item(),
            best_val_acc_graphsage.item()
        ))

################################ MATERIALIZZO

best_features_views = []
for label in graph.ntypes:
    best_features_views.append(f"{label.lower()}_view")

os.system(f"cd .\\feature_repo\\ && feast materialize-incremental -v {' -v '.join(best_features_views)} {datetime.now().isoformat()}")


################################ RECUPERO LE FEATURE MATERIALIZZATE

for label in graph.ntypes:
    #entity_rows – A list of dictionaries where each key-value is an entity-name, entity-value pair.
    neo4j_label_ids_number = [ node_labels_map_dgl_neo4j[label][x] for x in graph.nodes(label).numpy()]
    neo4j_label_ids_dict = [ {label.lower()+'_id_neo4j': x} for x in neo4j_label_ids_number]
    
    label_embedding_df = store.get_online_features(
        entity_rows=neo4j_label_ids_dict,
        features=[
            f"{label.lower()}_view:graphsage_embedding_{label.lower()}"
        ]
    ).to_df()

    label_embedding_df = label_embedding_df.set_index(label.lower()+'_id_neo4j').reindex(neo4j_label_ids_number)
    graph.nodes[label].data['embedding'] = torch.FloatTensor(np.stack(label_embedding_df[f'graphsage_embedding_{label.lower()}'].values, axis=0))
    

model = HeteroRGCN(graph, 256, 64, 4)
model.load_state_dict(best_modle_dict_graphsage)
model.eval()

test_acc = 0

logits = model(graph)
pred = logits.argmax(1)
test_acc = (pred[test] == test_labels).float().mean()
print( 'Best %.4f' % (
    test_acc
))


stop = time.time()
print("The time to create the graph:", stop - start)