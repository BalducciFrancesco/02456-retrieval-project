from src.retrievers.base_retriever import Retriever
from src.retrievers.models.encoder import TransformerEncoder

class DPR(Retriever):
    def __init__(self, cfg, documents: list[dict] = None, index_path: str = None) -> None:
        super().__init__(documents, index_path)
        self.cfg = cfg
        self.__load_models()
    
    def __load_models(self):
        # n_heads, n_layers, emb_dim, hidden_dim, feed_forward_dim
        query_encoder = TransformerEncoder(**self.cfg.model.query_encoder)
        document_encoder = TransformerEncoder(**self.cfg.model.document_encoder)
        
        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            query_encoder.load_state_dict(query_weights_path)

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if self.cfg.model.get('document_encoder_weights_path', None):
            document_encoder.load_state_dict(document_weights_path)

    def CalculateScores(self, query):
        raise NotImplementedError("Must overwrite")


        

