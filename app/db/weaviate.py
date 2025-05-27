"""
Weaviate vector database client for RAG pipeline.
"""
from functools import lru_cache
import weaviate
from weaviate.classes.init import Auth, AdditionalConfig
import weaviate.classes as wvc
from weaviate.collections.classes.config import (
    Configure,
    Property,
    DataType,
    Tokenization,
    VectorDistances, # Example import, if needed for specific distance metrics
)
from weaviate.exceptions import WeaviateQueryException, WeaviateStartUpError

from app.core.config import settings
from app.core.logging import logger


@lru_cache()
def get_weaviate_client():
    """
    Create and return a Weaviate client instance.
    Uses lru_cache to ensure only one client is created.
    """
    logger.debug("Attempting to create Weaviate client.")
    
    auth_credentials = None
    if settings.WEAVIATE_API_KEY:
        logger.info("WEAVIATE_API_KEY is set. Configuring Weaviate client with API key authentication.")
        auth_credentials = Auth.api_key(settings.WEAVIATE_API_KEY)
    else:
        logger.info("WEAVIATE_API_KEY is not set. Attempting anonymous connection to Weaviate.")

    logger.info(f"Attempting to connect to Weaviate at URL: {settings.WEAVIATE_URL}")

    try:
        # Configure timeouts using AdditionalConfig
        # (connect_timeout_seconds, read_timeout_seconds)
        additional_config = AdditionalConfig(timeout=(20, 60))

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=settings.WEAVIATE_URL,
            auth_credentials=auth_credentials,
            # skip_init_checks is True by default for connect_to_weaviate_cloud
            # and connect_to_local, connect_to_embedded.
            # It's False by default for connect_to_custom.
            # Explicitly setting it if it was intended to be non-default for cloud.
            skip_init_checks=True, # Revert to True to potentially speed up initial connection
                                   # If True, is_ready() might not give meaningful results initially.
            additional_config=additional_config
            # grpc_extra_options={"ssl": True} # Uncomment if using gRPC and SSL is required explicitly
        )
        
        # Perform a thorough health check.
        # is_ready() checks connectivity, schema, and module readiness.
        logger.debug("Performing connection readiness check...")
        if not client.is_ready():
            logger.error("Weaviate server is not ready. Please check Weaviate instance, network, schema, and module status.")
            # Attempt to get more specific error information if available
            try:
                status = client.cluster.get_nodes_status()
                logger.info(f"Weaviate node status: {status}")
            except Exception as status_e:
                logger.warning(f"Could not retrieve Weaviate node status: {status_e}")
            raise WeaviateStartUpError("Weaviate server is not ready. Full check failed.")

        logger.info("Successfully connected to Weaviate and server is ready.")
        return client
        
    except WeaviateStartUpError as e:
        logger.error(f"Failed to connect to Weaviate during startup: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while creating Weaviate client: {e}", exc_info=True)
        raise


def create_weaviate_schema():
    """
    Create the Weaviate schema for Task Copilot.
    This includes classes for Notion, GitHub, and Slack data.
    Ensures idempotency by checking if collections already exist.
    Uses Weaviate v4 configuration classes.
    """
    try:
        client = get_weaviate_client()
        if not client: # Should not happen if get_weaviate_client raises on failure
             logger.error("Failed to get Weaviate client for schema creation.")
             return
    except Exception as e:
        logger.error(f"Cannot proceed with schema creation, Weaviate client unavailable: {e}")
        return

    # Define common vectorizer configuration
    # Assuming 'text2vec-transformers' is the globally intended vectorizer for these collections.
    # If different vectorizers are needed per collection, this needs to be adjusted.
    common_vectorizer_config = Configure.Vectorizer.text2vec_transformers(
        vectorize_collection_name=False # Default v3 'vectorizeClassName': False
    )

    # Schema definitions using a more direct v4 approach
    # Note: 'dataType' in v3 was a list (e.g., ["text"]). In v4, it's a direct DataType enum.
    schemas_v4_definitions = [
        {
            "name": "NotionDocument",
            "description": "A document from Notion, including pages and databases",
            "vectorizer_config": common_vectorizer_config,
            "properties": [
                Property(name="title", description="The title of the Notion document", data_type=DataType.TEXT),
                Property(
                    name="content", 
                    description="The content of the Notion document", 
                    data_type=DataType.TEXT,
                    # v3: "skip": False, "vectorizePropertyName": False
                    # v4: default for skip_vectorization is False.
                    # v4: default for vectorize_property_name is True for text2vec-transformers.
                    # Explicitly setting vectorize_property_name=False if that was the v3 intent.
                    vectorize_property_name=False, 
                    tokenization=Tokenization.WORD # Default, can be specified if needed
                ),
                Property(
                    name="url", 
                    description="The URL of the Notion document", 
                    data_type=DataType.TEXT,
                    skip_vectorization=True # v3: "skip": True
                ),
                Property(
                    name="userId", 
                    description="ID of the user who owns this document", 
                    data_type=DataType.TEXT,
                    skip_vectorization=True # v3: "skip": True
                ),
                Property(
                    name="lastUpdated", 
                    description="When the document was last updated", 
                    data_type=DataType.DATE,
                    skip_vectorization=True # v3: "skip": True
                ),
            ]
        },
        {
            "name": "GitHubItem",
            "description": "An item from GitHub, such as issues, PRs, or files",
            "vectorizer_config": common_vectorizer_config,
            "properties": [
                Property(name="title", description="The title or name of the GitHub item", data_type=DataType.TEXT),
                Property(
                    name="content", 
                    description="The content of the GitHub item", 
                    data_type=DataType.TEXT,
                    vectorize_property_name=False # v3: "vectorizePropertyName": False
                ),
                Property(name="url", description="The URL of the GitHub item", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="type", description="The type of GitHub item (issue, PR, file)", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="userId", description="ID of the user who owns this item", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="lastUpdated", description="When the item was last updated", data_type=DataType.DATE, skip_vectorization=True),
            ]
        },
        {
            "name": "SlackMessage",
            "description": "A message from Slack",
            "vectorizer_config": common_vectorizer_config,
            "properties": [
                Property(name="channel", description="The Slack channel the message was posted in", data_type=DataType.TEXT, skip_vectorization=True),
                Property(
                    name="content", 
                    description="The content of the Slack message", 
                    data_type=DataType.TEXT,
                    vectorize_property_name=False # v3: "vectorizePropertyName": False
                ),
                Property(name="sender", description="The person who sent the message", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="timestamp", description="When the message was sent", data_type=DataType.DATE, skip_vectorization=True),
                Property(name="userId", description="ID of the user who owns this message data", data_type=DataType.TEXT, skip_vectorization=True),
            ]
        }
    ]

    for schema_def in schemas_v4_definitions:
        collection_name = schema_def["name"]
        try:
            if not client.collections.exists(collection_name):
                logger.info(f"Creating Weaviate collection: {collection_name} with vectorizer: {schema_def['vectorizer_config']}")
                client.collections.create(
                    name=collection_name,
                    description=schema_def.get("description"),
                    vectorizer_config=schema_def.get("vectorizer_config"),
                    properties=schema_def.get("properties"),
                    # Add other configurations like replication_config, sharding_config if needed
                    # e.g., replication_config=Configure.replication(factor=1)
                )
                logger.info(f"Successfully created Weaviate collection: {collection_name}")
            else:
                logger.info(f"Weaviate collection '{collection_name}' already exists. Skipping creation.")
                # Optionally, add logic here to update the collection if its configuration has changed.
                # This would involve comparing the existing config with the new schema_def and using client.collections.update()
                # For example:
                # current_config = client.collections.get(collection_name).config.get() # Get current config
                # if needs_update(current_config, schema_def):
                # client.collections.update(...)

        except WeaviateQueryException as qe:
            logger.error(f"Weaviate query error while creating collection {collection_name}: {qe}", exc_info=True)
        except Exception as e:
            logger.error(f"General error creating Weaviate collection {collection_name}: {e}", exc_info=True)

# Example of how you might call this, e.g., during application startup
# if __name__ == "__main__":
#     logger.info("Attempting to create Weaviate schema...")
#     create_weaviate_schema()
#     logger.info("Schema creation process finished.") 