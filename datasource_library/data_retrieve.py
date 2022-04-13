from feast import FeatureStore
import neo4j_datasource
import pandas as pd

# Inserisco i dati da neo4j a postgres

neo4j_datasource.run_retrieve_neo4j_db()


####################################################################################################################

store = FeatureStore(repo_path="./feature_repo")

################## PA:
feature_pa = [
    "paper_view:graphsage_embedding_paper",
    "paper_view:fastrp_embedding_paper",
    "paper_view:paper_id_neo4j",
    "pa_view:label(n)_pa",
    "pa_view:pa_id_neo4j",
    "pa_view:label(m)_pa",
    "author_view:author_id_neo4j",
    "author_view:fastrp_embedding_author",
    "author_view:graphsage_embedding_author",
]

query = f"SELECT \"paper_id_neo4j\", \"author_id_neo4j\", \"event_timestamp\" FROM \"PA\""

pa_df = store.get_historical_features(
    entity_df=query,
    features=feature_pa,
    full_feature_names=True
).to_df()

#print(pa_df.head())

################## PC:
feature_pc = [
    "paper_view:paper_id_neo4j",
    "pc_view:label(n)_pc",
    "pc_view:pc_id_neo4j",
    "pc_view:label(m)_pc",
    "conference_view:conference_id_neo4j",
    "conference_view:fastrp_embedding_conference",
    "conference_view:graphsage_embedding_conference",
]

query = f"SELECT \"paper_id_neo4j\", \"conference_id_neo4j\", \"event_timestamp\" FROM \"PC\""

pc_df = store.get_historical_features(
    entity_df=query,
    features=feature_pc,
    full_feature_names=True
).to_df()

#print(pc_df.head())

################## PT:
feature_pt = [
    "paper_view:paper_id_neo4j",
    "pt_view:label(n)_pt",
    "pt_view:pt_id_neo4j",
    "pt_view:label(m)_pt",
    "term_view:term_id_neo4j",
    "term_view:fastrp_embedding_term",
    "term_view:graphsage_embedding_term",
]

query = f"SELECT \"paper_id_neo4j\", \"term_id_neo4j\", \"event_timestamp\" FROM \"PT\""

pt_df = store.get_historical_features(
    entity_df=query,
    features=feature_pt,
    full_feature_names=True
).to_df()

################################ DATA MANIPULATION

def dict_col(df_input : pd.DataFrame) : 
    dict_col = {}
    for col in df_input.columns :
        if "paper_view__paper_id_neo4j" in col :
            dict_col.update({col: "start_id"})
        elif "author_view__author_id_neo4j" in col or "conference_view__conference_id_neo4j" in col or "term_view__term_id_neo4j" in col :
            dict_col.update({col: "end_id"})
        else :
            dict_col.update({col: col})
    return dict_col


dict_pa = dict_col(pa_df)
dict_pc = dict_col(pc_df)
dict_pt = dict_col(pt_df)
# Rinomino le colonne dei dataframe

pa_df.rename(dict_pa, axis=1, inplace=True)
pc_df.rename(dict_pc, axis=1, inplace=True)
pt_df.rename(dict_pt, axis=1, inplace=True)
