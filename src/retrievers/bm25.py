from scipy.sparse import lil_matrix
from src.retrievers.tf_idf import TFIDF
from src.data_utils.document import Document
import numpy as np
import math

class BM25(TFIDF):
    def __init__(self, documents: list[dict], index_path: str, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
        self.bm25_matrix = self.GetBM25Vector()
    
    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.PreprocessText(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        return document_lengths, average_document_length/len(self.index.GetDocuments())
    
    def GetBM25IDF(self):
        N = len(self.index.GetDocuments())
        return {key: math.log((N + 1) / (n_qi + 0.5)) for key, n_qi in self.GetDocumentFrequencies().items()}

    def GetBM25Vector(self):
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}
        docs = self.index.GetDocuments()
        bm25_vector = lil_matrix((len(self.index.GetDocuments()), len(self.corpus_vocabulary)),
                                  dtype=np.float32)
        bm25_idf = self.GetBM25IDF() # dict  of word: bm25_imf

        doc_lengths = [sum(list(self.GetDocumentTermCounts(doc).values())) for doc in docs]
        doc_total_length = sum(doc_lengths)
        avg_length = doc_total_length/len(doc_lengths)
        
        #IDF -> change to BM25
        for i, doc in enumerate(docs):
            for key, value in self.GetDocumentTermCounts(doc).items():
                bm25_vector[i, self.term_to_index[key]] = (value * (self.k1 + 1)) / (value + self.k1 * (1 - self.b + self.b * (doc_lengths[i]/avg_length))) * bm25_idf[key]

        return bm25_vector.tocsr()
    
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores