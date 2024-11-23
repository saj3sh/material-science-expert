from typing import List
from langchain_core.embeddings import Embeddings
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import torch

BATCH_SIZE = 16


class ChunkDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


class MatSciEmbeddings(Embeddings):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("m3rg-iitd/matscibert")
        self.model = AutoModel.from_pretrained("m3rg-iitd/matscibert")

    @staticmethod
    def __process_batch(batch_of_texts, model, tokenizer):
        inputs = tokenizer(
            batch_of_texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        # use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            embeddings = model(**inputs)[0].mean(dim=1)
        return embeddings.cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        dataset = ChunkDataset(texts)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        vector_embeddings = []
        for batch in dataloader:
            batch_embeddings = MatSciEmbeddings.__process_batch(
                [*batch], self._model, self._tokenizer)
            vector_embeddings.extend(batch_embeddings)
        return vector_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
