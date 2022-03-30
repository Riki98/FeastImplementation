import dgl
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any
from feast import FeatureStore
import dgl.data
from sklearn.model_selection import train_test_split
import pandas as pd
from rgcn_gnn import HeteroRGCN
import torch.nn.functional as F
import time
from driver_neo4j import run_query_return_data

def str_to_float_list(l) :
    tot_list = []
    for i in range(0, len(l)) : 
        s = l[i]
        f_list = []
        temp = ""
        for j in range(0, len(s)) :
            if s[j] != "{" and s[j] != "}" and s[j] != ",":
                temp = temp + s[j]
            if s[j] == "," or s[j] == "}":
                f_list.append(float(temp))
                temp = ""
        f_list = np.array(f_list)
        tot_list.append(f_list)
    return tot_list
    

####################################################################################################################################################

store = FeatureStore(repo_path="./feature_repo")

""" start = time.time()

################## PA:
feature_pa = [
    "paper_view:graphsage_embedding_paper",
    "paper_view:fastrp_embedding_paper",
    "paper_view:paper_id_neo4j",
    "pa_view:label(n)_pa",
    "pa_view:pa_id_neo4j",
    "pa_view:label(m)_pa",
    "author_view:author_id_neo4j",
    "author_view:fastrp_embedding_author",
    "author_view:graphsage_embedding_author",
]

query = f"SELECT \"paper_id_neo4j\", \"author_id_neo4j\", \"event_timestamp\" FROM \"PA\""

pa_df = store.get_historical_features(
    entity_df=query,
    features=feature_pa,
    full_feature_names=True
).to_df()

#print(pa_df.head())

################## PC:
feature_pc = [
    "paper_view:paper_id_neo4j",
    "pc_view:label(n)_pc",
    "pc_view:pc_id_neo4j",
    "pc_view:label(m)_pc",
    "conference_view:conference_id_neo4j",
    "conference_view:fastrp_embedding_conference",
    "conference_view:graphsage_embedding_conference",
]

query = f"SELECT \"paper_id_neo4j\", \"conference_id_neo4j\", \"event_timestamp\" FROM \"PC\""

pc_df = store.get_historical_features(
    entity_df=query,
    features=feature_pc,
    full_feature_names=True
).to_df()

#print(pc_df.head())

################## PT:
feature_pt = [
    "paper_view:paper_id_neo4j",
    "pt_view:label(n)_pt",
    "pt_view:pt_id_neo4j",
    "pt_view:label(m)_pt",
    "term_view:term_id_neo4j",
    "term_view:fastrp_embedding_term",
    "term_view:graphsage_embedding_term",
]

query = f"SELECT \"paper_id_neo4j\", \"PT\".\"term_id_neo4j\", \"PT\".\"event_timestamp\" FROM \"PT\", \"Term\" where \"PT\".\"term_id_neo4j\" = \"Term\".\"term_id_neo4j\" and \"Term\".\"term_term\" = 'management'"

pt_df = store.get_historical_features(
    entity_df=query,
    features=feature_pt,
    full_feature_names=True
).to_df()

#print(pt_df.head())


################################ DATA MANIPULATION

stop = time. time()
print("The time to retrieve:", stop - start)

start = time.time()

# creo la colonna has_term dove se il papar ha il termine "management" contiene 1, 0 altrimenti
pa_df["has_term"] = 0
for i in range(0, (len(pt_df.index))) :
    if pt_df.iloc[i]["paper_id_neo4j"] in pa_df :
        pa_df["has_term"][pt_df.iloc[i]["paper_id_neo4j"]] = 1

print("pa_df con has_term: ")
print(pa_df.head())

# converto gli embeddings da object in liste di float 
pa_df["author_view__graphsage_embedding_author"] = str_to_float_list(np.array(pa_df["author_view__graphsage_embedding_author"]))
pa_df["paper_view__graphsage_embedding_paper"] = str_to_float_list(np.array(pa_df["paper_view__graphsage_embedding_paper"]))
pc_df["conference_view__graphsage_embedding_conference"] = str_to_float_list(np.array(pc_df["conference_view__graphsage_embedding_conference"]))

 """

#########################################################################################################################################################

# Splitto il dataframe (80% Train (di cui 80% train 20% validate) e 20% test)

# x_train, x_test = train_test_split(X, test_size=0.2)
# x_train, x_validate = train_test_split(x_train, test_size = 0.2)

################################################################ COSTRUZIONE GRAFO ##################################################################

# stop = time. time()
# print("The time to create the graph:", stop - start)



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

""" dict_pa = dict_col(pa_df)
dict_pc = dict_col(pc_df)
dict_pt = dict_col(pt_df)
# Rinomino le colonne dei dataframe

pa_df.rename(dict_pa, axis=1, inplace=True)
pc_df.rename(dict_pc, axis=1, inplace=True)
pt_df.rename(dict_pt, axis=1, inplace=True) """

# g = g_load[(len(g_load) -1)]


################################################################# SALVATAGGIO DATAFRAME #################################################################
""" pa_df.to_pickle("pa_df.pkl")
pc_df.to_pickle("pc_df.pkl")
pt_df.to_pickle("pt_df.pkl") """

start = time.time()

################################################################# RECUPERO DATAFRAME RELAZIONI ####################################################################
pa_df = pd.read_pickle("pa_df.pkl")
pc_df = pd.read_pickle("pc_df.pkl")
pt_df = pd.read_pickle("pt_df.pkl")

###############################################################################################################################################

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
    label_embedding_df.to_pickle(f"{label.lower()}_graphsage.pkl")

    graph.nodes[label].data['embedding'] = torch.FloatTensor(np.stack(str_to_float_list(label_embedding_df[f'graphsage_embedding_{label.lower()}'].values), axis=0))


################################################################# SALVATAGGIO GRAFO #################################################################
# Per salvare il grafo, come ultimo parametro bisogna mettere anche un dizionario {'label': Tensor }
#dgl.save_graphs("DBLP_Graph", graph)

################################################################### RECUEPERO GRAFO ###################################################################
# Grafo con graphsage embeddings
# g_load, labels = dgl.load_graphs("DBLP_Graph")
# graph = g_load[0]
# print(graph)
# print(graph.ndata["Paper"])
#######################################################################################################################################################

import driver_neo4j

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

print(train_labels)
train_labels = torch.LongTensor(train_labels)
val_labels = torch.LongTensor(val_labels)
test_labels = torch.LongTensor(test_labels)


################ ALLENO IL MODELLO CON GRAPHSAGE

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


stop = time.time()
print("The time to create the graph:", stop - start)
