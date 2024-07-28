import os
from dotenv import load_dotenv

from aisearch.ctrl import index_ctrl

load_dotenv(override=True)  # take environment variables from .env.

# The following variables from your .env file are used in this notebook
endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
index_name = os.getenv("AZURE_SEARCH_INDEX", "vectest")
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_key = os.getenv("AZURE_OPENAI_KEY", None)
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
azure_openai_embedding_dimensions = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS"))
embedding_model_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")


def main():
    index_ctrl("text_sample.json", "docVectors.json")


if __name__ == "__main__":
    main()
