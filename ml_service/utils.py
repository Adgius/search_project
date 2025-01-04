import os
import json
import re
import nltk
import numpy as np
import torch
import logging

from typing import Dict, List, Tuple, Callable
from model import KNRM

logging.basicConfig(level=logging.INFO)

class MLService:
    def __init__(self,
                 knrm_embeddings_path: str = os.environ['EMB_PATH_KNRM'],
                 mlp_path: str = os.environ['MLP_PATH'],
                 vocab_path: str = os.environ['VOCAB_PATH'],
                 min_token_occurancies: int = 1,
                 random_seed: int = 0,
                 emb_rand_uni_bound: float = 0.2,
                 freeze_knrm_embeddings: bool = True,
                 knrm_kernel_num: int = 21,
                 knrm_out_mlp: List[int] = [],
                 dataloader_bs: int = 1024,
                 change_train_loader_ep: int = 10
                 ):
        self.knrm_embeddings_path = knrm_embeddings_path
        self.mlp_path = mlp_path
        self.vocab_path = vocab_path
        self.min_token_occurancies = min_token_occurancies
        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.change_train_loader_ep = change_train_loader_ep

        self.model, self.vocab = self.build_knrm_model()

    def handle_punctuation(self, inp_str: str) -> str:
        inp_str = re.sub(r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]""", ' ', inp_str)
        return inp_str

    def simple_preproc(self, inp_str: str) -> List[str]:
        inp_str = self.handle_punctuation(inp_str)
        inp_str = inp_str.lower()
        return nltk.word_tokenize(inp_str)

    def create_vocab_from_file(self, vocab_file_path: str) -> Dict[str, int]:
        with open(vocab_file_path, encoding='utf-8') as f:
            word2token = json.load(f)
        return word2token

    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        logger.info('Creating vocabulary and model ...')
        fitted_vocab = self.create_vocab_from_file(self.vocab_path)
        torch.manual_seed(self.random_seed)
        knrm = KNRM(embedding_path=self.knrm_embeddings_path,
                    mlp_path=self.mlp_path,
                    freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        logger.info('Created ...')
        return knrm, fitted_vocab

    def predict(self, dataloader):
        all_preds = []
        for batch in (dataloader):
            preds = self.model.predict(batch).squeeze(1)
            preds_np = preds.detach().numpy().tolist()
            all_preds.extend(preds_np)
        return all_preds


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, query: str, candidates: List,
                 vocab: Dict[str, int], oov_val: int,
                 preproc_func: Callable, max_len: int = 30):
        self.query = query
        self.candidates = {n: k for n, k in enumerate(candidates)}
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(candidates)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        res = [self.vocab.get(i, self.oov_val) for i in tokenized_text]
        return res

    def _convert_text_to_token_idxs(self, text: str) -> List[int]:
        tokenized_text = self.preproc_func(text)
        idxs = self._tokenized_text_to_index(tokenized_text)
        return idxs

    def __getitem__(self, idx: int):
        doc_label = self.candidates[idx]
        left_elem = {}
        left_elem['query'] = self._convert_text_to_token_idxs(self.query)
        left_elem['document'] = self._convert_text_to_token_idxs(self.candidates[idx])
        return left_elem
    
class CandidateModel:
    def __init__(self, es, index_name):
        self.es = es
        self.index_name = index_name

    def _fuzzy_search(self, query, size):
        body = {
            "size": size,
            "query": {
                "match": {
                    "question": {
                        "query": query,
                        "fuzziness": "AUTO"
                    }
                }
            }
        }
        response = self.es.search(index=self.index_name, body=body)
        return response["hits"]["hits"]

    def query(self, q, size=10):
        response = self._fuzzy_search(q, size=size)
        return [(i['_source']['index'], i['_source']['question']) for i in response]
    

def collate_fn(batch_objs: Dict[str, torch.Tensor]):
    max_len_q1 = -1
    max_len_d1 = -1

    is_triplets = False
    for elem in batch_objs:
        max_len_q1 = max(len(elem['query']), max_len_q1)
        max_len_d1 = max(len(elem['document']), max_len_d1)

    q1s = []
    d1s = []

    for elem in batch_objs:
        pad_len1 = max_len_q1 - len(elem['query'])
        pad_len2 = max_len_d1 - len(elem['document'])

        q1s.append(elem['query'] + [0] * pad_len1)
        d1s.append(elem['document'] + [0] * pad_len2)

    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)

    return {'query': q1s, 'document': d1s}