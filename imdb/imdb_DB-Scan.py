import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import pandas as pd
import imdb.imdb_neo4j_config as imdb_neo4j_config
from sklearn.manifold import TSNE


# #############################################################################
# Generate sample data
""" centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)
 """

actorName, act_embedded = imdb_neo4j_config.get_actors()
movieName, mov_embedded = imdb_neo4j_config.get_movies()
directorName, dir_embedded = imdb_neo4j_config.get_directors()

# Riduzione dimensionale con TSNE ----> ATTORI
act_tsne = TSNE(n_components=2, random_state=6).fit_transform(act_embedded)
# Riduzione dimensionale con TSNE ----> MOVIE
mov_tsne = TSNE(n_components=2, random_state=6).fit_transform(mov_embedded)
# Riduzione dimensionale con TSNE ----> DIRETTORI
dir_tsne = TSNE(n_components=2, random_state=6).fit_transform(dir_embedded)

df_actors = pd.DataFrame(data = {
    #"actor": actorName,
    "x": [value[0] for value in act_tsne],
    "y": [value[1] for value in act_tsne]
})

df_movies = pd.DataFrame(data = {
    #"movies": movieName,
    "x": [value[0] for value in mov_tsne],
    "y": [value[1] for value in mov_tsne]
})

df_directors = pd.DataFrame(data = {
    #"directors": directorName,
    "x": [value[0] for value in dir_tsne],
    "y": [value[1] for value in dir_tsne]
})


X = StandardScaler().fit_transform(df_directors)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)


# #############################################################################
# Plot result
# Black removed and is used for noise instead.
unique_labels = set(labels)
print(unique_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()
