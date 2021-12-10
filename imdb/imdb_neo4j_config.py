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

#######################################
# gn: Graph Name
# ps: Property String
# r: relationship
# l: list of

gn_Actor = "Actor"
gn_Keyword = "Keyword"
gn_Movie = "Movie"
gn_Director = "Director"

ps_id = "id"                                                    # stringa
ps_likes = "likes"
ps_embeddings = "embeddings"                                    # lista
ps_aspect_ratio = "aspect_ratio"
ps_budget = "budget"
ps_cast_tot_likes = "cast_total_facebook_likes"
ps_color = "color"                                              # stringa
ps_content_rating = "content_rating"
ps_duration = "duration"
ps_face_poster = "facenumber_in_poster"
ps_genre = "genre"
ps_gross = "gross"
ps_imdb_score = "imdb_score"
ps_language = "language"                                        # stringa
ps_movie_likes = "movie_facebook_likes"
ps_num_critics = "num_critic_for_reviews"
ps_user_for_review = "num_user_for_reviews"
ps_num_voted_users = "num_voted_users"
ps_title_year = "title_year"

r_performed = "PERFORMED_IN"
r_inplot = "IN_PLOT_OF"
r_directed = "DIRECTED_IN"

l_movie = {ps_aspect_ratio, ps_budget, ps_cast_tot_likes, ps_duration, ps_face_poster, ps_genre, ps_gross,
            ps_imdb_score, ps_movie_likes, ps_num_critics, ps_user_for_review, ps_num_voted_users, ps_title_year}
l_relationship = {r_performed, r_inplot, r_directed}

lista_movie = """'aspect_ratio', 'budget', 'cast_total_facebook_likes', 'duration', 
        'facenumber_in_poster', 'genre', 'gross', 'imdb_score', 'movie_facebook_likes', 'num_critic_for_reviews', 
        'num_user_for_reviews', 'num_voted_users', 'title_year'"""

driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "0000"))

#######################################

def get_actors():
    with driver.session() as session:
        result_act = session.run("""
        MATCH (p:Actor)
        WITH DISTINCT p
        RETURN p.id AS actorName, p.graphsage_embedding AS embeddingActor
        """)
        actorName = []
        embeddingsA = []
        for record in result_act:
            actorName.append(record["actorName"])
            embeddingsA.append(np.array(record["embeddingActor"]))
        embeddings_act = (np.stack(embeddingsA, 0))
        return (actorName, embeddings_act)

def get_movies():
    movieName = []
    embeddingsM = []
    with driver.session() as session:
        result_mov = session.run("""
        MATCH (p:Movie)
        WITH DISTINCT p
        RETURN p.id AS movieName, p.graphsage_embedding AS embeddingMovie
        """)
        for record in result_mov:
            embeddingsM.append(np.array(record["embeddingMovie"]))
            movieName.append(record["movieName"])
        embeddings_mov = (np.stack(embeddingsM, 0))
    return (movieName, embeddings_mov)

def get_directors() :
    directorName = []
    embeddingsD = []
    with driver.session() as session: 
        result_dir = session.run("""
        MATCH (p:Director)
        WITH DISTINCT p
        RETURN p.id AS directorName, p.graphsage_embedding AS embeddingDirector
        """)
        for record in result_dir:
            embeddingsD.append(np.array(record["embeddingDirector"]))
            directorName.append(record["directorName"])
        embeddings_dir = (np.stack(embeddingsD, 0))
    return (directorName, embeddings_dir)




# Import and plot degli embedding
def read_graph():
    with driver.session() as session:
        actorName, embeddings_act = get_actors()
        movieName, embeddings_mov = get_movies()
        directorName, embeddings_dir = get_directors()
        
        # Riduzione dimensionale con TSNE ----> ATTORI
        act_embedded = TSNE(n_components=2, random_state=6).fit_transform(embeddings_act)

        # Riduzione dimensionale con TSNE ----> FILM
        mov_embedded = TSNE(n_components=2, random_state=6).fit_transform(embeddings_mov)

        # Riduzione dimensionale con TSNE ----> DIRETTORI
        dir_embedded = TSNE(n_components=2, random_state=6).fit_transform(embeddings_dir)

        #plot
        figure1 = plt.scatter(act_embedded[:,0], act_embedded[:,1], color="green", label="Attori")
        figure2 = plt.scatter(mov_embedded[:,0], mov_embedded[:,1], color="red", label="Movie")
        firgure3 = plt.scatter(dir_embedded[:,0], dir_embedded[:,1], color="blue", label="Direttori")
        plt.legend()
        plt.show()


# FUNZIONA Eliminazione della proiezione del grafo in memoria
def delete_graph_projection(graph_name):
    with driver.session as session:
        drop = f"CALL gds.graph.drop('{graph_name}')"
        result = session.run(drop)
        for line in result:
            print(line)




################################# GRAPH SAGE (gds v. 1.5) #######################################
# FUNZIONA PER GRAPHSAGE
# 
# Creazione di una proiezione di un grafo
# controlla come sostituire le parentesi graffe con le quadre nella stringa della query
# name_proj, graph_name, property_list, relationship_list
# 
def create_graph_projection_graphsage(tx):
    with driver.session() as session:
        create = ("""CALL gds.graph.create('dbProjection', {
    Actor: {label: 'Actor', 
        properties: { 
            act_likes:{ property: 'likes', defaultValue:0.0 },    
            budget:{property:'budget', defaultValue:0.0},
            cast_total_facebook_likes:{property:'cast_total_facebook_likes', defaultValue:0.0},
            duration:{property:'duration', defaultValue:0.0},   
            facenumber_in_poster:{property:'facenumber_in_poster', defaultValue:0.0},   
            genre:{property:'genre', defaultValue:0.0},   
            gross:{property:'gross', defaultValue:0.0},   
            imdb_score:{property:'imdb_score', defaultValue:0.0},   
            movie_facebook_likes:{property:'movie_facebook_likes', defaultValue:0.0},   
            num_critic_for_reviews:{property:'num_critic_for_reviews', defaultValue:0.0},   
            num_user_for_reviews:{property:'num_user_for_reviews', defaultValue:0.0},   
            num_voted_users:{property:'num_voted_users', defaultValue:0.0},   
            title_year:{property:'title_year', defaultValue:0.0},
            dir_likes:{ property:'likes', defaultValue:0.0 }
        }  
    }, 
    Movie:{label: 'Movie', 
        properties: { 
            act_likes:{ property: 'likes', defaultValue:0.0 }, 
            budget:{property:'budget', defaultValue:0.0},
            cast_total_facebook_likes:{property:'cast_total_facebook_likes', defaultValue:0.0},
            duration:{property:'duration', defaultValue:0.0},   
            facenumber_in_poster:{property:'facenumber_in_poster', defaultValue:0.0},   
            genre:{property:'genre', defaultValue:0.0},   
            gross:{property:'gross', defaultValue:0.0},   
            imdb_score:{property:'imdb_score', defaultValue:0.0},   
            movie_facebook_likes:{property:'movie_facebook_likes', defaultValue:0.0},   
            num_critic_for_reviews:{property:'num_critic_for_reviews', defaultValue:0.0},   
            num_user_for_reviews:{property:'num_user_for_reviews', defaultValue:0.0},   
            num_voted_users:{property:'num_voted_users', defaultValue:0.0},   
            title_year:{property:'title_year', defaultValue:0.0},
            dir_likes:{ property: 'likes', defaultValue:0.0 }
        } 
    }, 
    Director:{label: 'Director', 
        properties: { 
            act_likes:{ property: 'likes', defaultValue:0.0 }, 
            budget:{property:'budget', defaultValue:0.0},
            cast_total_facebook_likes:{property:'cast_total_facebook_likes', defaultValue:0.0},
            duration:{property:'duration', defaultValue:0.0},   
            facenumber_in_poster:{property:'facenumber_in_poster', defaultValue:0.0},   
            genre:{property:'genre', defaultValue:0.0},   
            gross:{property:'gross', defaultValue:0.0},   
            imdb_score:{property:'imdb_score', defaultValue:0.0},   
            movie_facebook_likes:{property:'movie_facebook_likes', defaultValue:0.0},   
            num_critic_for_reviews:{property:'num_critic_for_reviews', defaultValue:0.0},   
            num_user_for_reviews:{property:'num_user_for_reviews', defaultValue:0.0},   
            num_voted_users:{property:'num_voted_users', defaultValue:0.0},   
            title_year:{property:'title_year', defaultValue:0.0},
            dir_likes:{ property: 'likes', defaultValue:0.0 }
        } 
    }
},
['PERFORMED_IN', 'DIRECTED_IN']
) 
YIELD graphName, nodeCount, relationshipCount, createMillis 
RETURN graphName, nodeCount, relationshipCount, createMillis
                """)
        result = tx.run(create)



# NON FUNZIONA Train per graphSage
def train_graphSage():
    with driver.session() as session:
        q_train = """CALL gds.beta.graphSage.train('dbProjection', {modelName:'esempioTrainModel', featureProperties:['act_likes', 'budget', 'cast_total_facebook_likes', 'duration', 
        'genre', 'gross', 'imdb_score', 'movie_facebook_likes',
        'num_voted_users', 'title_year', 'dir_likes'], embeddingDimension:1024})"""
        result=session.run(q_train)
        print("TRAIN GRAPHSAGE ESEGUITO \n")
        print(result)
        print("\n\n")

# HA SENSO? Stream per graphSage
def stream_graphSage():
    with driver.session() as session:
        q_stream = "call gds.beta.graphSage.stream('dbProjection', {modelName:'esempioTrainModel'})"
        result=session.run(q_stream)
        print("STREAM GRAPHSAGE ESEGUITO \n")
        print(result)
        print("\n\n")

# FUNZIONA Write per graphSage
def write_graphSage():
    with driver.session() as session:
        q_write = """
            CALL gds.beta.graphSage.write('dbProjection', {writeProperty:'graphsage_embedding', modelName:'esempioTrainModel'})
        """
        result=session.run(q_write)
        print("WRITE GRAPHSAGE ESEGUITO \n")
        print(result.graph)
        print("\n\n")



###################################### FAST RP (gds v. 1.5)  ######################################
# FUNZIONA PER FASTRP
# 
# Creazione di una proiezione di un grafo
# controlla come sostituire le parentesi graffe con le quadre nella stringa della query
# name_proj, graph_name, property_list, relationship_list
# 
def create_graph_projection_fastrp(tx):
    with driver.session() as session:
        create = ("""CALL gds.graph.create('dbProjection', {
    Actor: {label: 'Actor', 
        properties: { 
            act_likes:{ property: 'likes', defaultValue:0.0 }, 
            aspect_ratio:{ property:'aspect_ratio', defaultValue:0.0  },    
            budget:{property:'budget', defaultValue:0.0},
            cast_total_facebook_likes:{property:'cast_total_facebook_likes', defaultValue:0.0},
            duration:{property:'duration', defaultValue:0.0},   
            facenumber_in_poster:{property:'facenumber_in_poster', defaultValue:0.0},   
            genre:{property:'genre', defaultValue:0.0},   
            gross:{property:'gross', defaultValue:0.0},   
            imdb_score:{property:'imdb_score', defaultValue:0.0},   
            movie_facebook_likes:{property:'movie_facebook_likes', defaultValue:0.0},   
            num_critic_for_reviews:{property:'num_critic_for_reviews', defaultValue:0.0},   
            num_user_for_reviews:{property:'num_user_for_reviews', defaultValue:0.0},   
            num_voted_users:{property:'num_voted_users', defaultValue:0.0},   
            title_year:{property:'title_year', defaultValue:0.0},
            dir_likes:{ property:'likes', defaultValue:0.0 }
        }  
    }, 
    Movie:{label: 'Movie', 
        properties: { 
            act_likes:{ property: 'likes', defaultValue:0.0 }, 
            aspect_ratio:{ property:'aspect_ratio', defaultValue:0.0  },    
            budget:{property:'budget', defaultValue:0.0},
            cast_total_facebook_likes:{property:'cast_total_facebook_likes', defaultValue:0.0},
            duration:{property:'duration', defaultValue:0.0},   
            facenumber_in_poster:{property:'facenumber_in_poster', defaultValue:0.0},   
            genre:{property:'genre', defaultValue:0.0},   
            gross:{property:'gross', defaultValue:0.0},   
            imdb_score:{property:'imdb_score', defaultValue:0.0},   
            movie_facebook_likes:{property:'movie_facebook_likes', defaultValue:0.0},   
            num_critic_for_reviews:{property:'num_critic_for_reviews', defaultValue:0.0},   
            num_user_for_reviews:{property:'num_user_for_reviews', defaultValue:0.0},   
            num_voted_users:{property:'num_voted_users', defaultValue:0.0},   
            title_year:{property:'title_year', defaultValue:0.0},
            dir_likes:{ property: 'likes', defaultValue:0.0 }
        } 
    }, 
    Director:{label: 'Director', 
        properties: { 
            act_likes:{ property: 'likes', defaultValue:0.0 }, 
            aspect_ratio:{ property:'aspect_ratio', defaultValue:0.0  },    
            budget:{property:'budget', defaultValue:0.0},
            cast_total_facebook_likes:{property:'cast_total_facebook_likes', defaultValue:0.0},
            duration:{property:'duration', defaultValue:0.0},   
            facenumber_in_poster:{property:'facenumber_in_poster', defaultValue:0.0},   
            genre:{property:'genre', defaultValue:0.0},   
            gross:{property:'gross', defaultValue:0.0},   
            imdb_score:{property:'imdb_score', defaultValue:0.0},   
            movie_facebook_likes:{property:'movie_facebook_likes', defaultValue:0.0},   
            num_critic_for_reviews:{property:'num_critic_for_reviews', defaultValue:0.0},   
            num_user_for_reviews:{property:'num_user_for_reviews', defaultValue:0.0},   
            num_voted_users:{property:'num_voted_users', defaultValue:0.0},   
            title_year:{property:'title_year', defaultValue:0.0},
            dir_likes:{ property: 'likes', defaultValue:0.0 }
        } 
    }
},
{DIRECTED_IN: {orientation: 'UNDIRECTED'}, 
PERFORMED_IN: {orientation: 'UNDIRECTED'}}
) 
YIELD graphName, nodeCount, relationshipCount, createMillis 
RETURN graphName, nodeCount, relationshipCount, createMillis
                """)
        result = tx.run(create)


# Stima della memoria per FastRP
def memory_estimation():
    with driver.session() as session:
        q_mem = """CALL gds.fastRP.stream.estimate('dbProjection', {embeddingDimension: 128})
                    YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory"""
        result = session.run(q_mem)
        for ris in result:
            print(ris)
        node_cnt, rel_cnt, b_min, b_min, req = 0
        

# Stream per FastRP
def stream_fastRP():
    with driver.session() as session:
        q_stream = "call gds.beta.graphSage.stream('dbProjection', {embeddingDimension:128})"
        result=session.run(q_stream)
        print("STREAM FASTRP ESEGUITO \n")
        print(result)
        print("\n\n")

# Write per FastRP
def write_fastRp():
    with driver.session() as session:
        q_write = """
            call gds.fastRP.write('dbProjection', {embeddingDimension:256, writeProperty:'fastrp_embedding'})
        """
        session.run(q_write)
        

################################ MAIN #################################

if __name__ == "__main__":
    with driver.session() as session:
        #session.read_transaction(create_graph_projection_graphsage)
        #train_graphSage()
        #stream_graphSage()
        #write_graphSage() 
        read_graph()

        #session.read_transaction(create_graph_projection_fastrp)
        #session.read_transaction(memory_estimation())
        #session.write_transaction(write_fastRp())
        #read_graph()
        
    driver.close()