import os
import shutil
import json
from datasets import Dataset, DatasetDict, load_from_disk
from src.data_utils.query import Query
from src.data_utils.download_retrieval_datasets import get_fiqa_dataset, get_squad_dataset

class Corpus:
    def __init__(self, dataset_name: str, base_dir: str):
        self.dataset_name = dataset_name
        if not os.path.exists(f'{base_dir}/{self.dataset_name}'):
            raw_data_dir = f'{base_dir}/raw'
            os.makedirs(raw_data_dir, exist_ok=True)
            # load and process data
            self.ds = self.__load_data(base_dir=raw_data_dir)
            self.__rename_columns()
            self.relevant_ids = self.__assign_ids()
            os.makedirs(f'{base_dir}/{self.dataset_name}', exist_ok=True)
            DatasetDict(self.ds).save_to_disk(f'{base_dir}/{self.dataset_name}')
            with open(f'{base_dir}/{self.dataset_name}/relevant_ids.json', 'w') as f:
                json.dump(self.relevant_ids, f)
            
            print(f'Saved {self.dataset_name} dataset to path {base_dir}/{self.dataset_name}.')
            shutil.rmtree(f'{base_dir}/raw')
        else:
            self.ds = load_from_disk(f'{base_dir}/{self.dataset_name}')
            with open(f'{base_dir}/{self.dataset_name}/relevant_ids.json', 'r') as f:
                self.relevant_ids = json.load(f)


    @property
    def train(self):
        return self.ds.get('train', None)
    
    @property
    def val(self):
        return self.ds.get('validation', None)

    @property
    def test(self):
        return self.ds.get('test', None)

    @property
    def all_documents(self):
        # one entry per document id: a passage shared by several queries must appear in the index only once
        docs = {}
        for split in (self.train, self.val, self.test):
            if split is not None:
                for texts, ids in zip(split['text'], split['document_id']):
                    docs.update(zip(ids, texts))
        return Dataset.from_dict({'text': [[t] for t in docs.values()],
                                  'document_id': [[i] for i in docs]})

    def GetQuerySplits(self):
        def make(split):
            if split is None:
                return []
            return [
                Query(q, id=qid, relevant_document_ids=self.relevant_ids[qid])
                for q, qid in zip(split["question"], split["query_id"])
            ]
        return make(self.train), make(self.val), make(self.test)


    def __load_data(self, base_dir) -> dict[Dataset]:
        assert self.dataset_name in ['squad', 'fiqa'], f'Dataset {self.dataset_name} not supported. Options are ["squad", "fiqa"]'
        
        if self.dataset_name == 'fiqa':
            ds = get_fiqa_dataset(base_dir)

        elif self.dataset_name == 'squad':
            ds = get_squad_dataset(base_dir)

        return {key: ds[key] for key in ds.keys()}

    def __rename_columns(self) -> dict[Dataset]:
        for key in self.ds:
            if self.dataset_name == 'fiqa':
                self.ds[key] = self.ds[key].rename_column('ground_truths', 'text')
            else:  # squad: context is a single passage, wrap it so 'text' is always a list of passages
                self.ds[key] = self.ds[key].map(lambda row: {'text': [row['context']]}, remove_columns=['context'])
        return self.ds

    def __assign_ids(self):
        q_id_counter = 0
        doc_ids = {}  # passage text -> id, so the same passage always gets the same id
        relevant_document_ids = {}

        for key in self.ds:
            query_ids = []
            document_ids = []
            for documents in self.ds[key]['text']:  # read the column once, not once per row
                query_ids.append(f'q{q_id_counter}')
                relevant_ids = [doc_ids.setdefault(doc, f'doc{len(doc_ids)}') for doc in documents]
                document_ids.append(relevant_ids)
                relevant_document_ids[f'q{q_id_counter}'] = list(dict.fromkeys(relevant_ids))  # drop duplicates
                q_id_counter += 1

            self.ds[key] = self.ds[key].add_column('query_id', query_ids)
            self.ds[key] = self.ds[key].add_column('document_id', document_ids)

        return relevant_document_ids
            