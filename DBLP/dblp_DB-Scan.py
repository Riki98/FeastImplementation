import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import dblp_neo4j_config
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


# #############################################################################
# Generate sample data

#authName, auth_embedded = dblp_neo4j_config.get_authors()
#confName, conf_embedded = dblp_neo4j_config.get_conference()
paperName, paper_embedded = dblp_neo4j_config.get_paper()

# Riduzione dimensionale con TSNE ----> ATTORI
#auth_tsne = TSNE(n_components=3, random_state=6).fit_transform(auth_embedded)
# Riduzione dimensionale con TSNE ----> MOVIE
#conf_tsne = TSNE(n_components=2, random_state=6).fit_transform(conf_embedded)
# Riduzione dimensionale con TSNE ----> DIRETTORI
paper_tsne = TSNE(n_components=3, random_state=6).fit_transform(paper_embedded)

""" df_authors = pd.DataFrame(data = {
    #"actor": actorName,
    "x": [value[0] for value in auth_tsne],
    "y": [value[1] for value in auth_tsne], 
    "z": [value[2] for value in auth_tsne]
}) """

""" df_conf = pd.DataFrame(data = {
    #"movies": movieName,
    "x": [value[0] for value in conf_tsne],
    "y": [value[1] for value in conf_tsne]
})

df_paper = pd.DataFrame(data = {
    #"directors": directorName,
    "x": [value[0] for value in paper_tsne],
    "y": [value[1] for value in paper_tsne]
}) """


""" X = StandardScaler().fit_transform(df_authors) """

# #############################################################################
# Compute DBSCAN
""" db = DBSCAN(eps=0.3, min_samples=10).fit(df_authors)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1) """


# #############################################################################
# Plot result
# Black removed and is used for noise instead.
""" unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = df_authors[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = df_authors[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show() """



##################################################################################################################
# DBScan in 3d

model = DBSCAN(eps=2.5, min_samples=2)
model.fit_predict(paper_tsne)
pred = model.fit_predict(paper_tsne)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(paper_tsne[:,0], paper_tsne[:,1], paper_tsne[:,2], c=paper_tsne[:,2])
plt.show()