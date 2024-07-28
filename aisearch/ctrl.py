from aisearch.index_define import IndexDefine
import os
from dotenv import load_dotenv

load_dotenv(override=True)  # take environment variables from .env.

ENDPOINT = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX", "vectest")
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", None)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS"))
EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


def index_ctrl(input_name: str, output_name: str):
    # indexの定義、フィールド作成
    index_define = IndexDefine(
        azure_openai_key=AZURE_OPENAI_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_embedding_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        azure_openai_embedding_dimensions=AZURE_OPENAI_EMBEDDING_DIMENSIONS,
        endpoint=ENDPOINT,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        index_name=INDEX_NAME,
    )
    index_define.create_or_update_index()

    # データの作成、アップロード
    index_define.data_create(input_name, output_name)
    index_define.upload_documents(output_name)
