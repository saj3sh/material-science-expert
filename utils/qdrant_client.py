from qdrant_client import QdrantClient
import config


def get_qdrant_client():
    print(f'config.USE_LOCAL_QDRANT: {config.USE_LOCAL_QDRANT}')
    if config.USE_LOCAL_QDRANT:
        return QdrantClient(
            host="qdrant",
            port=6333,
            timeout=15
        )
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_TOKEN,
        port=6333,
        timeout=15
    )
