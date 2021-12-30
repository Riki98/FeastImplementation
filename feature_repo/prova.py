import numpy as np
import pandas as pd
import driver_neo4j
import neo4j_datasource

###################################################################################

# FUNZIONA PER GRAPHSAGE
# 
# Creazione di una proiezione di un grafo
# controlla come sostituire le parentesi graffe con le quadre nella stringa della query
# name_proj, graph_name, property_list, relationship_list
# 
def create_graph_projection_graphsage(tx, session):
    with driver_neo4j.session() as session:
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


## fai un unica tabella
def get_authors():
    res = driver_neo4j.run_transaction_query("""
        MATCH (p:Author)
        WITH DISTINCT p
        RETURN p.id AS authorId, p.graphsage_embedding AS embeddingAuthor
        """)
    authorId = []
    embeddingAuthor = []
    for dict in res["data"]:
        authorId.append(dict["authorId"])
        embeddingAuthor.append(dict["embeddingAuthor"])
    return (authorId, embeddingAuthor)
    



def get_conference():
    with driver_neo4j.session() as session:
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
    with driver_neo4j.session() as session:
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



############################################################ QUI IMPORTA GLI EMBEDDING DA NEO4J ##################################################################
""" auth_id, auth_emb = neo4j_graph_sage.get_authors()
timestamp_auth = [pd.Timestamp.now() for emb in auth_emb]
df_auth = pd.DataFrame(index=auth_id, columns=["Authors_embedding"])
df_auth["Authors_embedding"] = [np.array(emb) for emb in auth_emb]
df_auth["event_timestamp"] = timestamp_auth
df_auth["created"] = timestamp_auth """




############################################################ CREAZIONE TABELLE SU POSTGRES ##################################################################

""" table = "authors"
con.execute("DROP TABLE IF EXISTS " + table)
df_auth.to_sql(table, con, dtype={"Authors_embedding": ARRAY(FLOAT), "event_timestamp": TIMESTAMP, "created": TIMESTAMP})
res = con.execute("SELECT * FROM authors").fetchone()
print(type(res["Authors_embedding"])) """


