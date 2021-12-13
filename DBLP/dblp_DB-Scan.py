import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import dblp_neo4j_config
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import HDBSCAN


# #############################################################################
# Generate sample data

authId, auth_embedded = dblp_neo4j_config.get_authors()
#confName, conf_embedded = dblp_neo4j_config.get_conference()
#paperName, paper_embedded = dblp_neo4j_config.get_paper()

# Riduzione dimensionale con TSNE ----> Autori
#auth_tsne = TSNE(n_components=2, random_state=6).fit_transform(auth_embedded)
# Riduzione dimensionale con TSNE ----> Conferenze
#conf_tsne = TSNE(n_components=2, random_state=6).fit_transform(conf_embedded)
# Riduzione dimensionale con TSNE ----> Paper
#paper_tsne = TSNE(n_components=2, random_state=6).fit_transform(paper_embedded)

# controllare range graphsage e fastrp
#x = StandardScaler().fit_transform(df_authors)

auth_emb = np.array(auth_embedded)
auth_id = np.array(authId)

#controllo se ho tante colonne duplicate - axis = 0 -> riga
unq, index_inverse, count = np.unique(auth_emb, axis=0, return_counts=True, return_inverse=True)
#print(unq[count==count.max()])
count_max = unq[count==count.max()][0]
#print(count_max)

# vseleziono i 12 autori con lo stesso embedding
# ['A11208', 'A26999', 'A27289', 'A27609', 'A48747', 'A123329', 'A123394', 'A123588', 'A123870', 'A316459', 'A316460', 'A316461']
indice_count_max = np.all(auth_emb==count_max, axis=1)
#print(auth_emb[indice_count_max])
#print(auth_id[indice_count_max])


# #############################################################################
# Compute DBSCAN
model = DBSCAN(eps=1.65, min_samples=10).fit(auth_emb)
pred = model.fit(auth_emb)
labels = model.labels_
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[model.core_sample_indices_] = True

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)


# #############################################################################
# Plot result
# Black removed and is used for noise instead.
unique_labels = set(labels)
print(unique_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

auth_amb_reduced = TSNE(n_components=2, random_state=6).fit_transform(auth_emb)
print(zip(unique_labels, colors))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    # plotto i dati clusterizzati
    plt.plot(auth_amb_reduced[:, 0][labels == k], auth_amb_reduced[:, 1][labels == k], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=14)
    print("cluster numero: " + str(k))
    print("grandezza cluster: " + str((labels == k).sum()))

print(auth_id[labels == 1])


plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()

