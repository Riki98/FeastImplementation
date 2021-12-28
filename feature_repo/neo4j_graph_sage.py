from neo4j import GraphDatabase
import numpy as np
import pandas as pd
import driver_neo4j

###################################################################################

# FUNZIONA PER GRAPHSAGE
# 
# Creazione di una proiezione di un grafo
# controlla come sostituire le parentesi graffe con le quadre nella stringa della query
# name_proj, graph_name, property_list, relationship_list
# 
def create_graph_projection_graphsage(tx, session):
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



def prova(): 
    query = """match (p:Author) where p.id = "A582" return p.fastrp_embedding, p.graphsage_embedding, p.provaStringa, p.provaBoolean """
    return driver_neo4j.run_transaction_query(query, run_query=driver_neo4j.run_query_return_data_key)