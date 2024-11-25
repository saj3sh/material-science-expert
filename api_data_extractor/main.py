from datetime import datetime
import time
import uuid
from dotenv import load_dotenv
from mp_api.client import MPRester
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from utils.doc_formatter import format_summary_doc
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils.normalize_text import normalize
from torch.utils.data import DataLoader
from utils.custom_embeddings import MatSciEmbeddings
import config

if (input('WARNING: This operation will overwrite all existing embeddings. '
          'This means all previously stored vector embeddings and their '
          'associated data will be permanently deleted and replaced with new embeddings. \n\n'
          'Are you sure you want to proceed? (Y/N): ').strip().lower() != "y"):
    print("Operation aborted. No changes have been made.")
    exit(0)

qdrant_client = QdrantClient(
    host=config.QDRANT_URL, api_key=config.QDRANT_TOKEN, port=6333)

MATERIALS_COLLECTION_NAME = "materials"

if qdrant_client.collection_exists(collection_name=MATERIALS_COLLECTION_NAME):
    qdrant_client.delete_collection(collection_name=MATERIALS_COLLECTION_NAME)

qdrant_client.create_collection(
    collection_name=MATERIALS_COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)
# document model not used because of bug in API
# https://github.com/materialsproject/api/issues/922
with MPRester(config.MATERIAL_PROJECT_TOKEN) as mpr:
    fields_to_include = set(
        mpr.materials.summary.available_fields) - {"builder_meta", "last_updated", "origins"}
    material_docs = mpr.materials.summary.search(
        all_fields=False, fields=[*fields_to_include], chunk_size=48, num_chunks=1)
print("completed downloading documents from the Material Project API")
start_time = time.perf_counter()
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

embeddings_generator = MatSciEmbeddings(
).stream_embeddings_in_batch(description_chunks, batch_size=16)
for batch_start_idx, batch_end_idx, batch_embeddings in embeddings_generator:
    batch_material_ids = material_ids[batch_start_idx:batch_end_idx]
    batch_description_chunks = description_chunks[batch_start_idx:batch_end_idx]
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "material_id": material_id,
                "page_content": description_chunk
            }
        ) for embedding, material_id, description_chunk in zip(batch_embeddings, batch_material_ids, batch_description_chunks)
    ]
    # Add document to Qdrant
    qdrant_client.upsert(
        collection_name=MATERIALS_COLLECTION_NAME,
        points=points
    )
    print(f"{datetime.now()}: {len(points)} points inserted to Qdrant.")
end_time = time.perf_counter()
time_taken = end_time - start_time
print(f"Processing took {time_taken:.4f} seconds")
print(123)
