from feast import ValueType
from feast.data_source import DataSource
from feast.protos.feast.core.DataSource_pb2 import DataSource as DataSourceProto
from typing import Callable, Dict, Optional
from feast.repo_config import RepoConfig
import json
import driver_neo4j

class Neo4jSource(DataSource):
    def __init__(
        self,
        query: str,
        event_timestamp_column: Optional[str] = "",
        created_timestamp_column: Optional[str] = "",
        field_mapping: Optional[Dict[str, str]] = None,
        date_partition_column: Optional[str] = "",
    ):
        self._neo4j_options = Neo4jOptions(query=query)
        super().__init__(
            event_timestamp_column,
            created_timestamp_column,
            field_mapping,
            date_partition_column,
        )
    
    def __eq__(self, other):
        if not isinstance(other, Neo4jSource):
            raise TypeError(
                "Comparisons should only involve Neo4jSource class objects."
            )

        return (
            self._neo4j_options._query == other._neo4j_options._query
            and self.event_timestamp_column == other.event_timestamp_column
            and self.created_timestamp_column == other.created_timestamp_column
            and self.field_mapping == other.field_mapping
        )

    @staticmethod
    def from_proto(data_source: DataSourceProto):
        assert data_source.HasField("custom_options")

        _neo4j_options = json.loads(data_source.custom_options.configuration)
        return Neo4jSource(
            query=_neo4j_options["query"],
            field_mapping=dict(data_source.field_mapping),
            event_timestamp_column=data_source.event_timestamp_column,
            created_timestamp_column=data_source.created_timestamp_column,
            date_partition_column=data_source.date_partition_column,
        )

    def to_proto(self) -> DataSourceProto:
        data_source_proto = DataSourceProto(
            type=DataSourceProto.CUSTOM_SOURCE,
            field_mapping=self.field_mapping,
            custom_options=self._neo4j_options.to_proto(),
        )

        data_source_proto.event_timestamp_column = self.event_timestamp_column
        data_source_proto.created_timestamp_column = self.created_timestamp_column
        data_source_proto.date_partition_column = self.date_partition_column

        return data_source_proto

    def validate(self, config: RepoConfig):
        pass

    @staticmethod
    def source_datatype_to_feast_value_type() -> Callable[[str], ValueType]:
        return driver_neo4j.neo4j_type_to_feast_value_type()




class Neo4jOptions:
    def __init__(self, query: Optional[str]):
        self._query = query

    @classmethod
    def from_proto(cls, neo4j_options_proto: DataSourceProto.CustomSourceOptions):
        config = json.loads(neo4j_options_proto.configuration.decode("utf8"))
        neo4j_options = cls(query=config["query"],)

        return neo4j_options

    def to_proto(self) -> DataSourceProto.CustomSourceOptions:
        neo4j_options_proto = DataSourceProto.CustomSourceOptions(
            configuration=json.dumps({"query": self._query}).encode()
        )

        return neo4j_options_proto