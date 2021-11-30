from neo4j import GraphDatabase

#######################################
# gn: Graph Name
# ps: Property String
# r: relationship
# l: list of

gn_Actor = "Actor"
gn_Keyword = "Keyword"
gn_Movie = "Movie"
gn_Director = "Director"

ps_id = "id"
ps_likes = "likes"
ps_embeddings = "embeddings"
ps_aspect_ratio = "aspect_ratio"
ps_budget = "budget"
ps_cast_tot_likes = "cast_total_facebook_likes"
ps_color = "color"
ps_content_rating = "content_rating"
ps_duration = "duration"
ps_face_poster = "facenumber_in_poster"
ps_genre = "genre"
ps_gross = "gross"
ps_imdb_score = "imdb_score"
ps_language = "language"
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
        create = (f"""CALL gds.graph.create('actorMovieProjection', {{
                Actor: {{label: 'Actor', properties:{{ likes:{{ property: 'likes', defaultValue:0.0 }}, 
                    aspect_ratio:{{ property:'aspect_ratio', defaultValue:0.0  }}, 
                    budget:{{property:'budget', defaultValue:0}}   }} 
                    }}, 
                Movie:{{label: 'Movie', properties: {{likes:{{property: 'likes', defaultValue:0.0}}, 
                    aspect_ratio:{{property:'aspect_ratio', defaultValue:0.0}}, 
                    budget:{{property:'budget', defaultValue:0}}  }} 
                    }}, ['PERFORMED_IN']) 
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
        q_train = f"CALL gds.beta.graphSage.train('actorMovieProjection', {{modelName:'esempioTrainModel', featureProperty:['likes', 'aspect_property', 'budget']}})"
        result=session.run(q_train)
        print("TRAIN GRAPHSAGE ESEGUITO \n")
        print(result)
        print("\n\n\n\n")

# Stream per graphSage
def stream_graphSage():
    with driver.session() as session:
        q_stream = f"CALL gds.beta.graphSage.stream('actorProjection', {{modelName:'esempioTrainModel'}})"
        result=session.run(q_stream)
        print("STREAM GRAPHSAGE ESEGUITO \n")
        print(result)
        print("\n\n\n\n")

# Write per graphSage
def write_graphSage():
    with driver.session() as session:
        q_write = f"""
            CALL gds.beta.graphSage.train('actorProjection', {{modelName:'esempioTrainModel', degreeAsProperty:true}})
        """
        result=session.run(q_write)
        print("WRITE GRAPHSAGE ESEGUITO \n")
        print(result.graph)
        print("\n\n\n\n")



create_graph_projection()
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