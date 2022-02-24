import dgl
import numpy as np
import torch
from typing import Dict, List, Any
from feast import FeatureStore
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import dgl.data

import time

start = time.time()

""" store = FeatureStore(repo_path="./feature_repo")

################## PA:
feature_pa = [
    "papers_view:paper_id_neo4j",
    "pa_view:label(n)_pa",
    "pa_view:pa_id_neo4j",
    "pa_view:label(m)_pa",
    "authors_view:author_id_neo4j",
]

query = f"SELECT \"paper_id_neo4j\", \"author_id_neo4j\", \"event_timestamp\" FROM \"PA\""

pa_df = store.get_historical_features(
    entity_df=query,
    features=feature_pa,
    full_feature_names=True
).to_df()

print(pa_df.head())

################## PC:
feature_pc = [
    "papers_view:paper_id_neo4j",
    "pc_view:label(n)_pc",
    "pc_view:pc_id_neo4j",
    "pc_view:label(m)_pc",
    "conference_view:conference_id_neo4j",
]

query = f"SELECT \"paper_id_neo4j\", \"conference_id_neo4j\", \"event_timestamp\" FROM \"PC\""

pc_df = store.get_historical_features(
    entity_df=query,
    features=feature_pc,
    full_feature_names=True
).to_df()

print(pc_df.head())

################## PT:
feature_pt = [
    "papers_view:paper_id_neo4j",
    "pt_view:label(n)_pt",
    "pt_view:pt_id_neo4j",
    "pt_view:label(m)_pt",
    "term_view:term_id_neo4j",
]

query = f"SELECT \"paper_id_neo4j\", \"term_id_neo4j\", \"event_timestamp\" FROM \"PT\""

pt_df = store.get_historical_features(
    entity_df=query,
    features=feature_pt,
    full_feature_names=True
).to_df()

print(pt_df.head()) """


################################ GRAFI OMOGENEI
""" 
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
print("pa_df[\"label_paper\"]")
print(pa_df["papers_view__label_paper"].head())
print(f"len of papers_view__label_paper: { (len(pa_df['papers_view__label_paper'])) }")
print(f"len of papers_view__label_paper: {len(array_n)}")


######### NODES
g.ndata["papers_view__label_paper"] = torch.from_numpy(np.array(pa_df["papers_view__label_paper"]))

######### EDGES
g.edata["paper_id_neo4j"] = torch.from_numpy(np.array(pa_df["papers_view__paper_id_neo4j"]))
g.edata[""] = torch.from_numpy(np.array(pa_df["papers_view__label_paper"])) """


################################ GRAFI ETEROGENEI -- USA QUESTI

""" stop = time. time()
print("The time to retrieve:", stop - start)

start = time.time()

paper_id_tensor = torch.from_numpy(np.array(pa_df['papers_view__paper_id_neo4j']))

g = dgl.heterograph({
    ('paper', 'pa', 'author') : (paper_id_tensor, torch.from_numpy(np.array(pa_df['authors_view__author_id_neo4j']))),
    ('paper', 'pc', 'conference') : (paper_id_tensor, torch.from_numpy(np.array(pc_df['conference_view__conference_id_neo4j']))), 
    ('paper', 'pt', 'term') : (paper_id_tensor, torch.from_numpy(np.array(pt_df['term_view__term_id_neo4j'])))
    })

print(g)

dgl.save_graphs("DBLP_Graph", g)

stop = time. time()
print("The time to create the graph:", stop - start) """

############################### TRAIN 

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(100):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

g = dgl.load_graphs("DBLP_Graph")
print(g)

dataset = dgl.data.CoraGraphDataset() #va cambiato con il dataset del mio grafo
print(type(dataset))
""" model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
train(g, model) """

