import uuid
from mp_api.client import MPRester
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from doc_formatter import format_summary_doc
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from normalize_text import normalize
from torch.utils.data import DataLoader
from custom_embeddings import MatSciEmbeddings
import torch

if (input('WARNING: This operation will overwrite all existing embeddings. '
          'This means all previously stored vector embeddings and their '
          'associated data will be permanently deleted and replaced with new embeddings. \n\n'
          'Are you sure you want to proceed? (Y/N): ').strip().lower() != "y"):
    print("Operation aborted. No changes have been made.")
    exit(0)

# Initialize Qdrant client
# Update with your Qdrant instance details
qdrant_client = QdrantClient(host="localhost", port=6333)

MATERIALS_COLLECTION_NAME = "materials"

# Drop the collection if it already exists
if qdrant_client.collection_exists(collection_name=MATERIALS_COLLECTION_NAME):
    qdrant_client.delete_collection(collection_name=MATERIALS_COLLECTION_NAME)

# Create or update a collection in Qdrant
qdrant_client.create_collection(
    collection_name=MATERIALS_COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# document model not used because of bug in API
# https://github.com/materialsproject/api/issues/922
with MPRester("M4aseyAs7ose2uflpjD5iERCLf8eDjsS") as mpr:
    fields_to_include = set(
        mpr.materials.summary.available_fields) - {"builder_meta", "last_updated", "origins"}
    material_docs = mpr.materials.summary.search(
        all_fields=False, fields=[*fields_to_include])
print("completed downloading documents from the Material Project API")
material_descriptions = [format_summary_doc(doc) for doc in material_docs]
material_descriptions = [
    (material_id, normalize(material_description))
    for material_id, material_description in material_descriptions
]
token_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200)

chunked_documents = [
    (material_id, description_chunk)
    for material_id, material_description in material_descriptions
    for description_chunk in token_splitter.split_text(material_description)
]
material_ids, description_chunks = zip(*chunked_documents)

vector_embeddings = MatSciEmbeddings().embed_documents(description_chunks)

points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=vector_embedding,
        payload={
            "material_id": material_id,
            "description": description_chunk
        }
    ) for vector_embedding, material_id, description_chunk in zip(vector_embeddings, material_ids, description_chunks)
]
# Add document to Qdrant
qdrant_client.upsert(
    collection_name=MATERIALS_COLLECTION_NAME,
    points=points
)
print(123)
