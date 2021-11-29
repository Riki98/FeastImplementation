from neo4j import GraphDatabase

driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "0000"))

#def add_friend(tx, name, friend_name):
#    tx.run("MERGE (a:Person {name: $name}) "
#           "MERGE (a)-[:KNOWS]->(friend:Person {name: $friend_name})",
#           name=name, friend_name=friend_name)

#def print_friends(tx, name):
#    for record in tx.run("MATCH (a:Person)-[:KNOWS]->(friend) WHERE a.name = $name "
#                         "RETURN friend.name ORDER BY friend.name", name=name):
#        print(record["friend.name"])

#with driver.session() as session:
#    session.write_transaction(add_friend, "Arthur", "Guinevere")
#    session.write_transaction(add_friend, "Arthur", "Lancelot")
#    session.write_transaction(add_friend, "Arthur", "Merlin")
#    session.read_transaction(print_friends, "Arthur")


# FUNZIONA Creazionde di una proiezione di un grafo
def create_graph_projection(tx, name):
    create = """
        CALL gds.graph.create('$name', {Actor: {properties: 'likes'}, Movie: {properties: 'budget'} }, ['PERFORMED_IN'])
        YIELD graphName, nodeCount, relationshipCount, createMillis
        RETURN graphName, nodeCount, relationshipCount, createMillis
    """

    result = tx.run(create)
    for line in result:
        print(line)

# Eliminazione della proiezione del grafo in memoria
def delete_graph(tx, name):
    drop = """
        CALL gds.graph.drop('$name')
    """
    result = tx.run(drop)
    for line in result:
        print(line)


# Train per graphSage
def train_graphSage(tx):
    q_train = """
        CALL gds.beta.graphSage.train($nome, {modelName:'esempioTrainModel', degreeAsProperty:true})
    """
    result=tx.run(q_train)

# Stream per graphSage
def stream_graphSage(tx):
    q_stream = """
        CALL gds.beta.graphSage.stream($nome, {modelName:'esempioTrainModel'})
    """
    result=tx.run(q_stream)

# Write per graphSage
def write_graphSage(tx):
    q_write = """
        CALL gds.beta.graphSage.train($nome, {modelName:'esempioTrainModel', degreeAsProperty:true})
    """
    result=tx.run(q_write)



def prova(tx):
    persone = []
    ris = tx.run("MATCH (a:Person) RETURN a.name")
    for result in ris:
        persone.append(result)
    return persone


with driver.session() as session:
    name = 'grafoProva'
    res = session.read_transaction(prova)
    
    for persone in res:
        print(type(res))


driver.close()