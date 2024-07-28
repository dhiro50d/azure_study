import os
from pydantic import BaseModel

from openai import AzureOpenAI
import json
from azure.core.credentials import AzureKeyCredential

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex,
    AzureOpenAIVectorizer,
    AzureOpenAIParameters,
)


class IndexDefine(BaseModel):
    azure_openai_key: str
    azure_openai_endpoint: str
    azure_openai_embedding_deployment: str
    azure_openai_embedding_dimensions: int
    endpoint: str
    embedding_model_name: str
    index_name: str
    index_client: SearchIndexClient = None
    client: AzureOpenAI = None
    credential: AzureKeyCredential = AzureKeyCredential()
    search_client: SearchClient = None

    def _post_init_(self):
        openai_credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            openai_credential, "https://cognitiveservices.azure.com/.default"
        )
        self.client = AzureOpenAI(
            azure_deployment=self.azure_openai_embedding_deployment,
            api_version=self.azure_openai_api_version,
            azure_endpoint=self.azure_openai_endpoint,
            api_key=self.azure_openai_key,
            azure_ad_token_provider=(
                token_provider if not self.azure_openai_key else None
            ),
        )
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential,
        )

        self.index_client = SearchIndexClient(
            endpoint=self.endpoint, credential=self.credential
        )

    def data_create(self, fname, output_path):
        with open(fname, "r", encoding="utf-8") as file:
            input_data = json.load(file)

        titles = [item["title"] for item in input_data]
        content = [item["content"] for item in input_data]
        title_response = self.client.embeddings.create(
            input=titles,
            model=self.embedding_model_name,
            dimensions=self.azure_openai_embedding_dimensions,
        )
        title_embeddings = [item.embedding for item in title_response.data]
        content_response = self.client.embeddings.create(
            input=content,
            model=self.embedding_model_name,
            dimensions=self.azure_openai_embedding_dimensions,
        )
        content_embeddings = [item.embedding for item in content_response.data]

        # Generate embeddings for title and content fields
        for i, item in enumerate(input_data):
            title = item["title"]
            content = item["content"]
            item["titleVector"] = title_embeddings[i]
            item["contentVector"] = content_embeddings[i]

        # Output embeddings to docVectors.json file
        output_directory = os.path.dirname(output_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        with open(output_path, "w") as f:
            json.dump(input_data, f)
        return

    @property
    def fields(self):
        _fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                sortable=True,
                filterable=True,
                facetable=True,
            ),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(
                name="category", type=SearchFieldDataType.String, filterable=True
            ),
            SearchField(
                name="titleVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.azure_openai_embedding_dimensions,
                vector_search_profile_name="myHnswProfile",
            ),
            SearchField(
                name="contentVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.azure_openai_embedding_dimensions,
                vector_search_profile_name="myHnswProfile",
            ),
        ]
        return _fields

    @property
    def vector_search(self):
        """Configure the vector search configuration"""
        _vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                    vectorizer="myVectorizer",
                )
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    name="myVectorizer",
                    azure_open_ai_parameters=AzureOpenAIParameters(
                        resource_uri=self.azure_openai_endpoint,
                        deployment_id=self.azure_openai_embedding_deployment,
                        model_name=self.embedding_model_name,
                        api_key=self.azure_openai_key,
                    ),
                )
            ],
        )
        return _vector_search

    @property
    def semantic_config(self):
        _semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                keywords_fields=[SemanticField(field_name="category")],
                content_fields=[SemanticField(field_name="content")],
            ),
        )
        return _semantic_config

    def create_or_update_index(self):
        # Create the semantic settings with the configuration
        semantic_search = SemanticSearch(configurations=[self.semantic_config])

        # Create the search index with the semantic settings
        index = SearchIndex(
            name=self.index_name,
            fields=self.fields,
            vector_search=self.vector_search,
            semantic_search=semantic_search,
        )
        result = self.index_client.create_or_update_index(index)
        print(f" {result.name} created")

    def upload_documents(self, output_path: str):
        output_directory = os.path.dirname(output_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        with open(output_path, "r") as file:
            documents = json.load(file)
        search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential,
        )
        result = search_client.upload_documents(documents)
        print(f"Uploaded {len(documents)} documents")
        return
