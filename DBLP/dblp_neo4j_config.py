from neo4j import GraphDatabase
from neo4j.work import result
from numpy.core.fromnumeric import shape
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits import mplot3d

#######################################

driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "0000"))

#######################################

def get_authors():
    with driver.session() as session:
        result_auth = session.run("""
        MATCH (p:Author)
        WITH DISTINCT p
        RETURN p.id AS authorId, p.graphsage_embedding AS embeddingAuthor
        """)
        authorId = []
        embeddingAuthor = []
        for record in result_auth:
            authorId.append(record["authorId"])
            embeddingAuthor.append(record["embeddingAuthor"])
        return (authorId, embeddingAuthor)

def get_conference():
    with driver.session() as session:
        result_conf = session.run("""
        MATCH (p:Conference)
        WITH DISTINCT p
        RETURN p.id AS confId, p.graphsage_embedding AS embeddingConf
        """)
        confId = []
        embeddingConf = []
        for record in result_conf:
            confId.append(record["confId"])
            embeddingConf.append(np.array(record["embeddingConf"]))
        embeddings_conf = (np.stack(embeddingConf, 0))
        return (confId, embeddings_conf)

def get_paper():
    with driver.session() as session:
        result_paper = session.run("""
        MATCH (p:Paper)
        WITH DISTINCT p
        RETURN p.id AS paperId, p.graphsage_embedding AS embeddingPaper
        """)
        paperId = []
        embeddingPaper = []
        for record in result_paper:
            paperId.append(record["paperId"])
            embeddingPaper.append(np.array(record["embeddingPaper"]))
        embeddings_paper = (np.stack(embeddingPaper, 0))
        return (paperId, embeddings_paper)




# Import and plot degli embedding
def read_graph():
    with driver.session() as session:
        actorName, embeddings_auth = get_authors()
        movieName, embeddings_conf = get_conference()
        directorName, embeddings_paper = get_paper()
        
        # Riduzione dimensionale con TSNE ----> AUTORI
        #auth_embedded = TSNE(n_components=3, random_state=6).fit_transform(embeddings_auth)

        # Riduzione dimensionale con TSNE ----> CONFERENZA
        #conf_embedded = TSNE(n_components=3, random_state=6).fit_transform(embeddings_conf)

        # Riduzione dimensionale con TSNE ----> PAPER
        #paper_embedded = TSNE(n_components=3, random_state=6).fit_transform(embeddings_paper)

        #plot
        #figure1 = plt.scatter(auth_embedded[:,0], auth_embedded[:,1], color="green", label="Author")
        #figure2 = plt.scatter(conf_embedded[:,0], conf_embedded[:,1], color="red", label="Conference")
        #firgure3 = plt.scatter(paper_embedded[:,0], paper_embedded[:,1], color="blue", label="Paper")

        #plotting in 3d
        ax = plt.axes(projection='3d')
        #ax.scatter3D(conf_embedded[:,0], conf_embedded[:,1], conf_embedded[:,2], c=conf_embedded[:,2], cmap='Greens')
        #ax.scatter3D(auth_embedded[:,0], auth_embedded[:,1], auth_embedded[:,2], c=auth_embedded[:,2], cmap='Reds')
        #ax.scatter3D(paper_embedded[:,0], paper_embedded[:,1], paper_embedded[:,2], c=paper_embedded[:,2], cmap='Blues')

        plt.show()


# FUNZIONA Eliminazione della proiezione del grafo in memoria
def delete_graph_projection(graph_name):
    with driver.session as session:
        drop = f"CALL gds.graph.drop('{graph_name}')"
        result = session.run(drop)
        for line in result:
            print(line)




################################# GRAPH SAGE (gds v. 1.7.1) #######################################
# FUNZIONA PER GRAPHSAGE
# 
# Creazione di una proiezione di un grafo
# controlla come sostituire le parentesi graffe con le quadre nella stringa della query
# name_proj, graph_name, property_list, relationship_list
# 
def create_graph_projection_graphsage(tx):
    with driver.session() as session:
        create = ("""CALL gds.graph.create('dbProjection', {
    Conference: {label: 'Conference', 
        properties: { 
            conf_label:{ property: 'label', defaultValue:0.0 },    
            paper_label:{ property: 'label', defaultValue:0.0 },
            author_label:{ property: 'label', defaultValue:0.0 }
        }  
    }, 
    Paper:{label: 'Paper', 
        properties: { 
            conf_label:{ property: 'label', defaultValue:0.0 },    
            paper_label:{ property: 'label', defaultValue:0.0 },
            author_label:{ property: 'label', defaultValue:0.0 }
        } 
    }, 
    Author:{label: 'Author', 
        properties: { 
            conf_label:{ property: 'label', defaultValue:0.0 },    
            paper_label:{ property: 'label', defaultValue:0.0 },
            author_label:{ property: 'label', defaultValue:0.0 }
        } 
    }
},
['PC', 'PT', 'PA']
) 
YIELD graphName, nodeCount, relationshipCount, createMillis 
RETURN graphName, nodeCount, relationshipCount, createMillis
                """)
        result = tx.run(create)



# NON FUNZIONA Train per graphSage
def train_graphSage():
    with driver.session() as session:
        q_train = """CALL gds.beta.graphSage.train('dbProjection', {modelName:'dblp_model', featureProperties:['conf_label', 'paper_label', 'author_label'], embeddingDimension:256})"""
        result=session.run(q_train)
        print("TRAIN GRAPHSAGE ESEGUITO \n")
        print(result)
        print("\n\n")


# FUNZIONA Write per graphSage
def write_graphSage():
    with driver.session() as session:
        q_write = """
            CALL gds.beta.graphSage.write('dbProjection', {writeProperty:'graphsage_embedding', modelName:'dblp_model'})
        """
        result=session.run(q_write)
        print("WRITE GRAPHSAGE ESEGUITO \n")
        print(result.graph)
        print("\n\n")



###################################### FAST RP (gds v. 1.7.1)  ######################################
# FUNZIONA PER FASTRP
# 
# Creazione di una proiezione di un grafo
# controlla come sostituire le parentesi graffe con le quadre nella stringa della query
# name_proj, graph_name, property_list, relationship_list
# 
def create_graph_projection_fastrp(tx):
    with driver.session() as session:
        create = ("""CALL gds.graph.create('dbFastProjection', {
    Conference: {label: 'Conference', 
        properties: { 
            conf_label:{ property: 'label', defaultValue:0.0 },    
            paper_label:{ property: 'label', defaultValue:0.0 },
            author_label:{ property: 'label', defaultValue:0.0 }
        }  
    }, 
    Paper:{label: 'Paper', 
        properties: { 
            conf_label:{ property: 'label', defaultValue:0.0 },    
            paper_label:{ property: 'label', defaultValue:0.0 },
            author_label:{ property: 'label', defaultValue:0.0 }
        } 
    }, 
    Author:{label: 'Author', 
        properties: { 
            conf_label:{ property: 'label', defaultValue:0.0 },    
            paper_label:{ property: 'label', defaultValue:0.0 },
            author_label:{ property: 'label', defaultValue:0.0 }
        } 
    }
},
{PC: {orientation: 'UNDIRECTED'}, 
PT: {orientation: 'UNDIRECTED'},
PA: {orientation: 'UNDIRECTED'}}
) 
YIELD graphName, nodeCount, relationshipCount, createMillis 
RETURN graphName, nodeCount, relationshipCount, createMillis
                """)
        result = tx.run(create)


# Stima della memoria per FastRP
def memory_estimation():
    with driver.session() as session:
        q_mem = """CALL gds.fastRP.stream.estimate('dbFastProjection', {embeddingDimension: 128})
                    YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory"""
        result = session.run(q_mem)
        for ris in result:
            print(ris)
        node_cnt, rel_cnt, b_min, b_min, req = 0


# Write per FastRP
def write_fastRp():
    with driver.session() as session:
        q_write = """
            call gds.fastRP.write('dbFastProjection', {embeddingDimension:256, writeProperty:'fastrp_embedding'})
        """
        session.run(q_write)
        

################################ MAIN #################################

if __name__ == "__main__":
    with driver.session() as session:
        #session.read_transaction(create_graph_projection_graphsage)
        #train_graphSage()
        #stream_graphSage()
        #write_graphSage() 
        #read_graph()

        #session.read_transaction(create_graph_projection_fastrp)
        #session.read_transaction(memory_estimation())
        #session.write_transaction(write_fastRp)
        read_graph()
        
    driver.close()