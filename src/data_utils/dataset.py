import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from src.data_utils.document import Document

def batched(items: list, batch_size: int):
    """Yield consecutive chunks of `items`, each of length `batch_size` (the last one may be shorter)."""
    for i in tqdm(range(0, len(items), batch_size), desc="Batch"):
        yield items[i : i + batch_size]


class DocumentDataset:
    def __init__(self, documents: list[dict], embeddings: NDArray[np.float32] = None):
        self.documents = self.__BuildDocuments(documents)
        if embeddings is not None and embeddings.shape[0] != len(self.documents):
            raise ValueError(f"Embeddings shape {embeddings.shape} does not match number of documents {len(self.documents)}")
        self.embeddings = embeddings
        
    def __BuildDocuments(self, documents):
        return [Document(text=text, _id=_id) 
                for entries in documents 
                for (text, _id) in zip(entries['text'], entries['document_id'])
        ]
        
    def GetDocuments(self) -> list[Document]:
        return self.documents

    def GetEmbeddings(self) -> NDArray[np.float32]:
        return self.embeddings
