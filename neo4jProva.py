from neo4j import GraphDatabase
from sklearn.manifold import TSNE
import numpy as np
import altair as alt
import pandas as pd

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
def create_graph_projection():
    with driver.session() as session:
        create = (f"""CALL gds.graph.create('personMovieProjection', {{
                Person: {{label: 'Person', properties:{{ born:{{ property: 'born', defaultValue:0.0 }}, 
                    aspect_ratio:{{ property:'aspect_ratio', defaultValue:0.0  }}, 
                    budget:{{property:'budget', defaultValue:0}}   }} 
                    }}, 
                Movie:{{label: 'Movie', properties: {{born:{{property: 'born', defaultValue:0.0}}, 
                    aspect_ratio:{{property:'aspect_ratio', defaultValue:0.0}}, 
                    budget:{{property:'budget', defaultValue:0}}  }} 
                    }} }}, ['ACTED_IN']) 
            YIELD graphName, nodeCount, relationshipCount, createMillis 
            RETURN graphName, nodeCount, relationshipCount, createMillis""")
        result = session.run(create)
        for line in result:
            print(line)

# FUNZIONA Eliminazione della proiezione del grafo in memoria
def delete_graph_projection(graph_name):
    with driver.session as session:
        drop = f"CALL gds.graph.drop('{graph_name}')"
        result = session.run(drop)
        for line in result:
            print(line)

# Train per graphSage
# graph_name
def train_graphSage():
    with driver.session() as session:
        q_train = f"CALL gds.beta.graphSage.train('personMovieProjection', {{modelName:'esempioTrainModel', featureProperties:['likes', 'aspect_ratio', 'budget']}})"
        result=session.run(q_train)
        print("TRAIN GRAPHSAGE ESEGUITO \n")
        print(result)
        print("\n\n")

# Stream per graphSage
def stream_graphSage():
    with driver.session() as session:
        q_stream = f"CALL gds.beta.graphSage.stream('personMovieProjection', {{modelName:'esempioTrainModel'}})"
        result=session.run(q_stream)
        print("STREAM GRAPHSAGE ESEGUITO \n")
        print(result)
        print("\n\n")

# Write per graphSage
def write_graphSage():
    with driver.session() as session:
        q_write = f"""
            CALL gds.beta.graphSage.train('personMovieProjection', {{modelName:'esempioTrainModel'}})
        """
        result=session.run(q_write)
        print("WRITE GRAPHSAGE ESEGUITO \n")
        print(result.graph)
        print("\n\n")


def reading_datas():
    with driver.session() as session:
        result = session.run("""
        MATCH (p:Actor)-[:PERFORMED_IN]->(m:Movie)
        WHERE m.id = 'Pirates of the Caribbean: On Stranger Tides '
        RETURN p.id AS actorName, p.embedding AS embedding, m.id AS movieName
        """)
        actorName = []
        embeddings = []
        movieName = []
        for record in result:
            actorName.append(record["actorName"])
            embeddings.append(record["embedding"])
            movieName.append(record["movieName"])
        df_actors = pd.DataFrame(np.array([actorName, embeddings, movieName]), columns=["actorName", "embedding", "movieName"], dtype=object)
        x_embedded = TSNE(n_components=2, random_state=6).fit_transform(df_actors.embedding)
        print(x_embedded)
        #df = pd.DataFrame(data = {
        #    "actors": df_actors.actorName,
        #    "movies": df_actors.movieName,
        #    "x": [value[0] for value in x_embedded],
        #    "y": [value[1] for value in x_embedded]
        #})


reading_datas()



#create_graph_projection()
#train_graphSage()
#stream_graphSage()
#write_graphSage()



def read_graph(tx, graph_name):
    persone = []
    ris = tx.run(f"MATCH (a:{graph_name}) RETURN a.id")
    for result in ris:
        persone.append(result)
    return persone





driver.close()