from typing import Generator, List, Tuple
from langchain_core.embeddings import Embeddings
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
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
        self._tokenizer = AutoTokenizer.from_pretrained("m3rg-iitd/matscibert")
        self._model = AutoModel.from_pretrained("m3rg-iitd/matscibert")

    @staticmethod
    def __process_batch(batch_of_texts, model, tokenizer, print_device):
        inputs = tokenizer(
            batch_of_texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        # use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if print_device:
            print(f'the device is {device}')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            embeddings = model(**inputs)[0].mean(dim=1)
        return embeddings.cpu().tolist()

    def stream_embeddings_in_batch(self, texts: List[str], batch_size=3200) -> Generator[Tuple[int, int, List[float]], None, None]:
        assert batch_size % BATCH_SIZE == 0, f"batch_size should be a multiple of {BATCH_SIZE}"
        dataset = ChunkDataset(texts)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        acc_embeddings = []
        start_idx = 0
        print_device = True
        for batch in dataloader:
            embeddings = MatSciEmbeddings.__process_batch(
                [*batch], self._model, self._tokenizer, print_device
            )
            print_device = False
            acc_embeddings.extend(embeddings)
            while len(acc_embeddings) >= batch_size:
                end_idx = start_idx + batch_size   # exclusive
                print(f'{datetime.now()}: generated {batch_size} embeddings')
                yield start_idx, end_idx, acc_embeddings[:batch_size]
                acc_embeddings = acc_embeddings[batch_size:]
                start_idx = end_idx
        if acc_embeddings:
            yield start_idx, start_idx + len(acc_embeddings), acc_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [embedding for batch in self.stream_embeddings_in_batch(texts) for embedding in batch[2]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
