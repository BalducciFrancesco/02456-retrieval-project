
import hydra
import pandas as pd
from src.data_utils.corpus import Corpus
from src.retrievers.bm25 import BM25
from src.training.trainer import Trainer
from src.retrievers.tf_idf import TFIDF
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate


@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    train_queries, val_queries, test_queries = corpus.GetQuerySplits()
    index_path = cfg.retriever.index_path.format(dataset=cfg.dataset.name)

    ### Initialize the model
    match cfg.retriever.method:
        case "dpr":
            model = Trainer(cfg, corpus, val_queries).train()
        case "tf_idf":
            model = TFIDF(documents=corpus.all_documents, index_path=index_path)
        case "bm25":
            model = BM25(documents=corpus.all_documents, index_path=index_path)
        case _:
            raise ValueError(f"Unknown retriever: {cfg.retriever}")

    print("Model ready, evaluating now...")

    ### Retrieve top-k relevant queries
    train_relevancies = retrieveQueryAndGetRelevancies(model, train_queries, k=cfg.top_k, batch_size=cfg.batch_size)
    val_relevancies = retrieveQueryAndGetRelevancies(model, val_queries, k=cfg.top_k, batch_size=cfg.batch_size)
    test_relevancies = retrieveQueryAndGetRelevancies(model, test_queries, k=cfg.top_k, batch_size=cfg.batch_size)

    ### Evaluate
    train_scores = Evaluate(train_relevancies, train_queries, k=cfg.top_k, batch_size=cfg.batch_size)
    val_scores = Evaluate(val_relevancies, val_queries, k=cfg.top_k, batch_size=cfg.batch_size)
    test_scores = Evaluate(test_relevancies, test_queries, k=cfg.top_k, batch_size=cfg.batch_size)

    print(pd.DataFrame({'train': train_scores, 'val': val_scores, 'test': test_scores}).T)


if __name__ == '__main__':
    main()
