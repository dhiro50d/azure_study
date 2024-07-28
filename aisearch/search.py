from openai import AzureOpenAI

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from pydantic import BaseModel


class AiSearchOperator(BaseModel):
    azure_openai_embedding_deployment: str
    azure_openai_api_version: str
    azure_openai_endpoint: str
    azure_openai_key: str
    azure_openai_embedding_dimensions: int
    embedding_model_name: str
    endpoint: str
    index_name: str
    search_client: any = None
    client: any = None
    credential: any = None

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

    def vector_search(self, query: str, top_k: int = 3, filter: str = None):
        embedding = self.client.embeddings.create(
            input=query,
            model=self.embedding_model_name,
            dimensions=self.azure_openai_embedding_dimensions,
        )
        embedding = embedding.data[0].embedding
        vector_query = VectorizedQuery(
            vector=embedding, k_nearest_neighbors=top_k, fields="contentVector"
        )

        """vector_query can alternative below code."""
        # vector_query = VectorizableTextQuery(
        #     text=query, k_nearest_neighbors=top_k, fields="contentVector"
        # )

        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["title", "content", "category"],
            filter=filter,
        )

        # for result in results:
        #     print(f"Title: {result['title']}")
        #     print(f"Score: {result['@search.score']}")
        #     print(f"Content: {result['content']}")
        #     print(f"Category: {result['category']}\n")
        return results

    def knn_exact_search(self, query: str, top_k: int = 3):
        vector_query = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=top_k,
            fields="contentVector",
            exhaustive=True,
        )

        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["title", "content", "category"],
        )

        return results

    def cross_vector_search(self, query: str, top_k: int = 3):
        vector_query = VectorizableTextQuery(
            text=query, k_nearest_neighbors=top_k, fields="contentVector, titleVector"
        )

        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["title", "content", "category"],
        )
        return results

    def nulti_vector_search(
        self,
        query: str,
        top_k_1: int = 3,
        top_k_2: int = 3,
        weight_1: float = 2.0,
        weight_2: float = 0.5,
    ):
        vector_query_1 = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=top_k_1,
            fields="titleVector",
            weight=weight_1,
        )
        vector_query_2 = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=top_k_2,
            fields="contentVector",
            weight=weight_2,
        )

        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query_1, vector_query_2],
            select=["title", "content", "category"],
        )
        return results

    def hybrid_search(self, query: str, top_k: int = 3, weight: float = 0.2):
        vector_query = VectorizableTextQuery(
            text=query, k_nearest_neighbors=top_k, fields="contentVector", weight=weight
        )

        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=["title", "content", "category"],
            top=top_k,
        )
        return results

    def semantic_hybrid_search(self, query: str, top_k: int = 3):
        vector_query = VectorizableTextQuery(
            text=query, k_nearest_neighbors=3, fields="contentVector", exhaustive=True
        )
        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=["title", "content", "category"],
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="my-semantic-config",
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            top=top_k,
        )

        semantic_answers = results.get_answers()
        # print anser
        """
        for answer in semantic_answers:
            if answer.highlights:
                print(f"Semantic Answer: {answer.highlights}")
            else:
                print(f"Semantic Answer: {answer.text}")
            print(f"Semantic Answer Score: {answer.score}\n")

        for result in results:
            print(f"Title: {result['title']}")
            print(f"Reranker Score: {result['@search.reranker_score']}")
            print(f"Content: {result['content']}")
            print(f"Category: {result['category']}")

            captions = result["@search.captions"]
            if captions:
                caption = captions[0]
                if caption.highlights:
                    print(f"Caption: {caption.highlights}\n")
                else:
                    print(f"Caption: {caption.text}\n")
        """
        return semantic_answers
