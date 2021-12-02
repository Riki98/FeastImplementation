from neo4j import GraphDatabase
from sklearn.manifold import TSNE
import numpy as np
import altair as alt
import pandas as pd
import math
import matplotlib.pyplot as plt

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

lista_movie = "'aspect_ratio', 'budget', 'cast_total_facebook_likes', 'duration', \
        'facenumber_in_poster', 'genre', 'gross', 'imdb_score', 'movie_facebook_likes', 'num_critic_for_reviews', \
        'num_user_for_reviews', 'num_voted_users', 'title_year'"
#######################################



driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "0000"))


# Creazionde di una proiezione di un grafo
# controlla come sostituire le parentesi graffe con le quadre nella stringa della query
# name_proj, graph_name, property_list, relationship_list
def create_graph_projection(tx):
    with driver.session() as session:
        create = ("""CALL gds.graph.create('actorMovieProjection', {
                Person: {label: 'Actor', properties:{
                    born:{ property: 'likes', defaultValue:0.0 }, 
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
                    title_year:{property:'title_year', defaultValue:0.0}
                    } 
                    }, 
                Movie:{label: 'Movie', properties: { 
                    born:{ property: 'likes', defaultValue:0.0 }, 
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
                    title_year:{property:'title_year', defaultValue:0.0}  
                    } 
                    }
                    }, ['PERFORMED_IN']) 
                YIELD graphName, nodeCount, relationshipCount, createMillis 
                RETURN graphName, nodeCount, relationshipCount, createMillis
                """)
        result = tx.run(create)
        print(result)
        for line in result:
            print(line)

# FUNZIONA Eliminazione della proiezione del grafo in memoria
def delete_graph_projection(graph_name):
    with driver.session as session:
        drop = f"CALL gds.graph.drop('{graph_name}')"
        result = session.run(drop)
        for line in result:
            print(line)




##################### GRAPH SAGE (gds v. 1.5) ###########################
# Train per graphSage
def train_graphSage():
    with driver.session() as session:
        q_train = "CALL gds.beta.graphSage.train('actorMovieProjection', {modelName:'esempioTrainModel', featureProperties:['born', 'aspect_ratio', 'budget', 'cast_total_facebook_likes', 'duration', \
        'facenumber_in_poster', 'genre', 'gross', 'imdb_score', 'movie_facebook_likes', 'num_critic_for_reviews', \
        'num_user_for_reviews', 'num_voted_users', 'title_year'], embeddingDimension:10})"
        result=session.run(q_train)
        print("TRAIN GRAPHSAGE ESEGUITO \n")
        print(result)
        print("\n\n")

# Stream per graphSage
def stream_graphSage():
    with driver.session() as session:
        q_stream = "call gds.beta.graphSage.stream('actorMovieProjection', {modelName:'esempioTrainModel'})"
        result=session.run(q_stream)
        print("STREAM GRAPHSAGE ESEGUITO \n")
        print(result)
        print("\n\n")

# Write per graphSage
def write_graphSage():
    with driver.session() as session:
        q_write = """
            CALL gds.beta.graphSage.write('actorMovieProjection', {writeProperty:'embedding', modelName:'esempioTrainModel'})
        """
        result=session.run(q_write)
        print("WRITE GRAPHSAGE ESEGUITO \n")
        print(result.graph)
        print("\n\n")




########################## IMPORT AND SHOWING DATAS FROM NEO4J ##########################
def reading_datas():
    with driver.session() as session:
        result = session.run("""
        MATCH (p:Actor)-[:PERFORMED_IN]->(m:Movie)
        WHERE m.id = 'Pirates of the Caribbean: On Stranger Tides '
        RETURN p.id AS actorName, p.embedding AS embeddingActor, m.embedding as embeddingMovie, m.id AS movieName
        """)
        actorName = []
        embeddingsA = []
        embeddingsM = []
        movieName = []
        for record in result:
            actorName.append(record["actorName"])
            embeddingsA.append(np.array(record["embeddingActor"]))
            embeddingsM.append(np.array(record["embeddingMovie"]))
            movieName.append(record["movieName"])


        #creo dataframe completo di attori, film in cui hanno recitato e tutti e due gli ambeddings corrispondenti
        #df_actors_movie = pd.DataFrame.from_dict({"actorName": actorName, "embeddingsA": embeddingsA, "embeddingsM": embeddingsM, "movieName": movieName})
        #print(df_actors)

        #concateno le due liste di embeddings
        #x = np.concatenate((np.stack(df_actors_movie["embeddingsA"].to_numpy(), 0), np.stack(df_actors_movie["embeddingsM"].to_numpy(), 0)), 0)
        #print("\n\n")
        #print((np.stack(df_actors_movie["embeddingsA"].to_numpy(), 0)))

        #creo 
        embeddings_act = (np.stack(embeddingsA, 0))
        embeddings_mov = (np.stack(embeddingsM, 0))


        print("\n\nMovie name:")
        print(movieName)
        print("\n\n")

        # Riduzione dimensionale con TSNE ----> ATTORI
        act_embedded = TSNE(n_components=2, random_state=6).fit_transform(embeddings_act)
        print("\n\n")
        print("EMBEDDINGS ATTORI:")
        print(act_embedded)
        print("\n\n")

        # Riduzione dimensionale con TSNE ----> FILM
        mov_embedded = TSNE(n_components=2, random_state=6).fit_transform(embeddings_mov)
        print("\n\n")
        print("EMBEDDINGS FILM:")
        print(mov_embedded)
        print("\n\n")


        df = pd.DataFrame(data = {
            "actor": actorName,
            "movie": movieName,
            "x": [value[0] for value in act_embedded],
            "y": [value[1] for value in mov_embedded]
        })

        print("\n\nDATAFRAME ACTOR MOVIE")
        print(df)
        print("\n\n")

        plt.scatter(df.x, df.y)
        plt.show()

        #alt.Chart(df).mark_circle(size=60).encode(
            #x='x',
            #y='y',
            #color='Actor',
            #tooltip=['actor', 'movie']
        #).properties(width=700, height=400)

        

reading_datas()



#create_graph_projection()
#train_graphSage()
#stream_graphSage()
#write_graphSage()




driver.close()