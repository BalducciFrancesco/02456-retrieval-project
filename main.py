import hydra
from src.data_utils.corpus import Corpus
from src.retrievers.bm25 import BM25
from src.retrievers.dpr import DPR
from src.retrievers.tf_idf import TFIDF
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate
from src.training.trainer import Trainer

@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    train_queries, val_queries, test_queries = corpus.GetQuerySplits()

    ### Initialize the model
    match cfg.retriever.method:
        case "dpr":
            model = DPR(cfg, documents=corpus.all_documents)
            trainer = Trainer(corpus.train)
            trainer.train(model.query_encoder, model.document_encoder)
        case "tf-idf":
            model = TFIDF(documents=corpus.all_documents)
        case "bm25":
            model = BM25(documents=corpus.all_documents)
        case _:
            raise ValueError(f"Unknown retriever: {cfg.retriever}")


    ### Retrieve top-k relevant queries
    train_relevancies = retrieveQueryAndGetRelevancies(model, train_queries, k=cfg.top_k)
    val_relevancies = retrieveQueryAndGetRelevancies(model, val_queries, k=cfg.top_k)
    test_relevancies = retrieveQueryAndGetRelevancies(model, test_queries, k=cfg.top_k)

    ### Evaluate
    train_scores = Evaluate(train_relevancies, train_queries)
    val_scores = Evaluate(val_relevancies, val_queries)
    test_scores = Evaluate(test_relevancies, test_queries)

    print(f'Train results: {train_scores}')
    print(f'Val results: {val_scores}')
    print(f'Test results: {test_scores}')

if __name__ == '__main__':
    main()