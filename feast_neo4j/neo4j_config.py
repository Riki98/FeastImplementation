# must end with OfflineStore as suffix
class OfflineStore():
    # invoked when reading values from the offline store using the FeatureStore.get_historica_features() method.
    # Return a RetrievalJob
    def get_historical_features(self, config: RepoConfig, feature_views: List[FeatureView], feature_refs: List[str], entity_df: Union[pd.DataFrame, str], registry: Registry, 
                                project: str, full_feature_names: bool = False): # -> RetrievalJob:
        print("Getting historical features from my offline store")
        return super().get_historical_features(config, feature_views, feature_refs, entity_df, registry, project, full_feature_names)

    # invoked when running feast materialize or feast materialize-incremental commands, 
    # or FeatureStore.materialize() method.
    # Return a RetrievalJob
    def pull_latest_from_table_or_query(self, config: RepoConfig, data_source: DataSource, join_key_columns: List[str], feature_name_columns: List[str], event_timestamp_column: str, 
                                        created_timestamp_column: Optional[str], start_date: datetime, end_date: datetime): # -> RetrievalJob:
        print("Pulling latest features from my offline store")
        return super().pull_latest_from_table_or_query(config, data_source, join_key_columns, feature_name_columns, event_timestamp_column, created_timestamp_column, start_date, end_date)

class OfflineStoreConfig()

class RetrievalJob()

class DataSourtce()