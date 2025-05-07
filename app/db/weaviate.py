"""
Weaviate vector database client for RAG pipeline.
"""
from functools import lru_cache
import weaviate
from weaviate.auth import AuthApiKey

from app.core.config import settings
from app.core.logging import logger


@lru_cache()
def get_weaviate_client():
    """
    Create and return a Weaviate client instance.
    Uses lru_cache to ensure only one client is created.
    """
    try:
        logger.debug("Creating Weaviate client")
        
        # Configure auth if API key is provided
        auth_config = None
        if settings.WEAVIATE_API_KEY:
            auth_config = AuthApiKey(api_key=settings.WEAVIATE_API_KEY)
        
        # Create client
        client = weaviate.Client(
            url=settings.WEAVIATE_URL,
            auth_client_secret=auth_config,
        )
        
        # Verify connection
        if not client.is_ready():
            raise ConnectionError("Weaviate server is not ready")
        
        return client
    except Exception as e:
        logger.error(f"Failed to create Weaviate client: {e}", exc_info=True)
        # Re-raise to allow proper error handling at the API level
        raise


def create_weaviate_schema():
    """
    Create the Weaviate schema for Task Copilot.
    This includes classes for Notion, GitHub, and Slack data.
    """
    client = get_weaviate_client()
    
    # Define schemas for each data source
    schemas = [
        {
            "class": "NotionDocument",
            "description": "A document from Notion, including pages and databases",
            "vectorizer": "text2vec-transformers",
            "moduleConfig": {
                "text2vec-transformers": {
                    "vectorizeClassName": False
                }
            },
            "properties": [
                {
                    "name": "title",
                    "description": "The title of the Notion document",
                    "dataType": ["text"],
                },
                {
                    "name": "content",
                    "description": "The content of the Notion document",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                },
                {
                    "name": "url",
                    "description": "The URL of the Notion document",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "userId",
                    "description": "ID of the user who owns this document",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "lastUpdated",
                    "description": "When the document was last updated",
                    "dataType": ["date"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                }
            ]
        },
        {
            "class": "GitHubItem",
            "description": "An item from GitHub, such as issues, PRs, or files",
            "vectorizer": "text2vec-transformers",
            "moduleConfig": {
                "text2vec-transformers": {
                    "vectorizeClassName": False
                }
            },
            "properties": [
                {
                    "name": "title",
                    "description": "The title or name of the GitHub item",
                    "dataType": ["text"],
                },
                {
                    "name": "content",
                    "description": "The content of the GitHub item",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                },
                {
                    "name": "url",
                    "description": "The URL of the GitHub item",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "type",
                    "description": "The type of GitHub item (issue, PR, file)",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "userId",
                    "description": "ID of the user who owns this item",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "lastUpdated",
                    "description": "When the item was last updated",
                    "dataType": ["date"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                }
            ]
        },
        {
            "class": "SlackMessage",
            "description": "A message from Slack",
            "vectorizer": "text2vec-transformers",
            "moduleConfig": {
                "text2vec-transformers": {
                    "vectorizeClassName": False
                }
            },
            "properties": [
                {
                    "name": "channel",
                    "description": "The Slack channel the message was posted in",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "content",
                    "description": "The content of the Slack message",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                },
                {
                    "name": "sender",
                    "description": "The person who sent the message",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "timestamp",
                    "description": "When the message was sent",
                    "dataType": ["date"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "userId",
                    "description": "ID of the user who owns this message data",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                }
            ]
        }
    ]
    
    # Create each class in the schema
    for schema in schemas:
        class_name = schema["class"]
        try:
            # Check if class already exists
            if not client.schema.exists(class_name):
                client.schema.create_class(schema)
                logger.info(f"Created Weaviate class: {class_name}")
            else:
                logger.info(f"Weaviate class already exists: {class_name}")
        except Exception as e:
            logger.error(f"Error creating Weaviate class {class_name}: {e}", exc_info=True) 