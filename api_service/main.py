import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# os.environ['EMB_PATH_GLOVE'] = './glove.6B.50d.txt'
# os.environ['EMB_PATH_KNRM'] = './embeddings.bin'
# os.environ['MLP_PATH'] = './knrm_mlp.bin'
# os.environ['VOCAB_PATH'] = './vocab.json'


from typing import Dict, List, Tuple, Callable

import sys
import json
import re
import nltk
import numpy as np
import torch
# import faiss
import logging

logging.basicConfig(level=logging.INFO)

class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        transformation = -torch.pow((x - self.mu), 2) / (2 * self.sigma ** 2)
        return torch.exp(transformation)


class KNRM(torch.nn.Module):
    def __init__(self, embedding_path: str, mlp_path: str,
                 freeze_embeddings: bool = True,
                 kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()

        self.cosine_sim = torch.nn.CosineSimilarity(dim=3, eps=1e-6)
        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

        self._load_model(mlp_path, embedding_path, freeze_embeddings)

    def _load_model(self, mlp_path, emb_path, freeze_embeddings) -> None:
        embedding_matrix = torch.load(emb_path)
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix['weight']),
            freeze=freeze_embeddings,
            padding_idx=0
        )
        self.mlp.load_state_dict(torch.load(mlp_path))

    def _get_kernels_layers(self) -> torch.nn.ModuleList:

        def _get_kernels_mu(k):
            step = 1 / (k - 1)
            left = -1 + step
            right = 1 - step
            return np.hstack([np.arange(left, right, (right - left)/ (k - 2)), right, 1])

        kernels = torch.nn.ModuleList()
        mu = _get_kernels_mu(self.kernel_num)
        for m in mu:
            if m != 1:
                kernel = GaussianKernel(m, self.sigma)
            else:
                kernel = GaussianKernel(1, self.exact_sigma)
            kernels.append(kernel)
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        fnn_lst = []

        current_nn = self.kernel_num
        for l in self.out_layers:
            fnn_lst.append(torch.nn.ReLU())
            fnn_lst.append(torch.nn.Linear(current_nn, l))
            current_nn = l

        fnn_lst.append(torch.nn.ReLU())
        fnn_lst.append(torch.nn.Linear(current_nn, 1))

        return torch.nn.Sequential(*fnn_lst)

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        alpha = 1e-6
        query = self.embeddings(query)
        doc = self.embeddings(doc)
        nominator = (query.unsqueeze(dim=2) * doc.unsqueeze(dim=1)).sum(axis=3)
        denuminator = torch.sqrt(torch.sum(query * query, axis=2) + alpha).unsqueeze(dim=2) * torch.sqrt(torch.sum(doc * doc, axis=2) + alpha).unsqueeze(dim=1)
        matrix = nominator / (denuminator)
        return matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left], [Batch, Right]
        query, doc = inputs['query'], inputs['document']

        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out



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


class Project:
    def __init__(self,
                 glove_vectors_path: str = os.environ['EMB_PATH_GLOVE'],
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
        self.glove_vectors_path = glove_vectors_path
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

        self.model, self.glove_vocab, self.vocab = self.build_knrm_model()

    def handle_punctuation(self, inp_str: str) -> str:
        inp_str = re.sub(r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]""", ' ', inp_str)
        return inp_str

    def simple_preproc(self, inp_str: str) -> List[str]:
        inp_str = self.handle_punctuation(inp_str)
        inp_str = inp_str.lower()
        return nltk.word_tokenize(inp_str)

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        with open(file_path, encoding='utf-8') as f:
            emb = f.readlines()
            glove_vocab = set()
            for row in emb:
                glove_vocab.add(row.split()[0])
        return glove_vocab

    def create_vocab_from_file(self,
                                   glove_emb_file_path: str,
                                   vocab_file_path: str
                                   ) -> Tuple[np.ndarray, Dict[str, int]]:
        glove_vocab = self._read_glove_embeddings(glove_emb_file_path)
        with open(vocab_file_path, encoding='utf-8') as f:
            word2token = json.load(f)
        return glove_vocab, word2token

    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        # app.logger.info('Creating vocabulary and model ...')
        glove_vocab, fitted_vocab = self.create_vocab_from_file(
            self.glove_vectors_path, self.vocab_path)
        torch.manual_seed(self.random_seed)
        knrm = KNRM(embedding_path=self.knrm_embeddings_path,
                    mlp_path=self.mlp_path,
                    freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        # app.logger.info('Created ...')
        return knrm, glove_vocab, fitted_vocab

    def predict(self, dataloader):
        all_preds = []
        for batch in (dataloader):
            preds = self.model.predict(batch).squeeze(1)
            preds_np = preds.detach().numpy().tolist()
            all_preds.extend(preds_np)
        return all_preds

class Candidate_model():

    def __init__(self, main_class):
        self.main_class = main_class
        self.emb_dim = self.main_class.model.embeddings(torch.tensor(0)).shape[0]

    def _create_doc_embedding(self, d):
        tokens = self.main_class.simple_preproc(d)
        emb = torch.zeros(self.emb_dim)
        for t in tokens:
            if t in self.main_class.glove_vocab:
                emb += self.main_class.model.embeddings(torch.tensor(self.main_class.vocab[t]))
        return (emb / torch.sqrt(torch.sum(emb * emb))).numpy()

    def create_index(self, documents):
        # 1 / 0
        # app.logger.info('Creating indexing ...')
        global faiss_index, id_to_text
        faiss_index_2_original = {n: k for n, k in enumerate(documents.keys())}
        original_index_2_faiss = {v: k for k, v in faiss_index_2_original.items()}
        embs = np.empty((len(documents), self.emb_dim), dtype=np.float32)
        for n, doc in enumerate(documents.values()):
            embs[n] = self._create_doc_embedding(doc)
        index = faiss.IndexFlatL2(self.emb_dim)
        id_to_text = documents
        # index = faiss.index_factory(emb_dim, "IVF16384_HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
        # index.train(embs)
        index.add(embs)
        faiss_index = index
        self.original_index_2_faiss = original_index_2_faiss
        self.faiss_index_2_original = faiss_index_2_original
        # app.logger.info('Done')

    def get_candidates(self, q, n=31):
        global faiss_index, id_to_text
        D, I = faiss_index.search(self._create_doc_embedding(q).reshape(1, -1), n)
        output_idx = list(map(self.faiss_index_2_original.get, I[0, 1:]))
        return output_idx, [id_to_text[i] for i in output_idx]

class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, query: str, index_candidates: List,
                 idx_to_text_mapping: Dict[str, str],
                 vocab: Dict[str, int], oov_val: int,
                 preproc_func: Callable, max_len: int = 30):
        self.query = query
        self.index_candidates = {n: k for n, k in enumerate(index_candidates)}
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_candidates)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        res = [self.vocab.get(i, self.oov_val) for i in tokenized_text]
        return res

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        tokenized_text = self.preproc_func(self.idx_to_text_mapping[idx])
        idxs = self._tokenized_text_to_index(tokenized_text)
        return idxs

    def __getitem__(self, doc_idx: int):
        doc_label = self.index_candidates[doc_idx]
        left_elem = {}
        left_elem['query'] = self._tokenized_text_to_index(self.preproc_func(self.query))
        left_elem['document'] = self._convert_text_idx_to_token_idxs(doc_label)
        return left_elem

def pipeline(q, candidate_model):
    global id_to_text
    candidates, sentences = candidate_model.get_candidates(q, n=15)
    ds = RankingDataset(query=q,
                    index_candidates=candidates,
                    idx_to_text_mapping=id_to_text,
                    vocab=candidate_model.main_class.vocab,
                    oov_val=candidate_model.main_class.vocab['OOV'],
                    preproc_func=candidate_model.main_class.simple_preproc)
    dl = torch.utils.data.DataLoader(
            ds, batch_size=candidate_model.main_class.dataloader_bs, num_workers=0,
            collate_fn=collate_fn, shuffle=False)
    score = candidate_model.main_class.predict(dl)
    reranged_idx = np.argsort(score)[:-10:-1]
    final_idx = list(map(ds.index_candidates.get, reranged_idx))
    return final_idx
if __name__ == "__main__":
    import os
    import json
    from flask import Flask, request, jsonify
    from langdetect import detect
    import faiss
    import time
    # import threading

    # Инициализация Flask приложения
    app = Flask(__name__)

    # Глобальные переменные для хранения модели и индекса
    is_initialized = False
    faiss_index = None
    id_to_text = {}  # Для хранения id текста и соответствующего текста

    def initialize_models():
        """Фоновая функция для инициализации моделей."""
        global is_initialized, rerank_model, candidate_model
        rerank_model = Project()
        candidate_model = Candidate_model(rerank_model)
        is_initialized = True

    @app.route('/ping', methods=['GET'])
    def ping():
        """Проверка готовности сервера."""
        global is_initialized
        if is_initialized:
            return jsonify({"status": "ok"})
        # return jsonify({"status": "initializing"})

    @app.route('/update_index', methods=['POST'])
    def update_index():
        """Обработка данных для обновления индекса."""
        global candidate_model, faiss_index, id_to_text

        # try:
        content = json.loads(request.data)
        documents = content.get("documents", {})

        if not documents:
            return jsonify({"status": "error", "message": "No documents provided."}), 400

        # Создание индекса
        id_to_text = documents
        candidate_model.create_index(documents)

        return jsonify({"status": "ok", "index_size": len(documents)})

        # except Exception as e:
        #     return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/query', methods=['POST'])
    def query():
        """Поиск похожих вопросов."""
        global candidate_model, faiss_index, id_to_text

        if candidate_model is None or faiss_index is None:
            return jsonify({"status": "FAISS is not initialized!"}), 400

        try:
            content = json.loads(request.data)
            queries = content.get("queries", [])

            if not queries:
                return jsonify({"status": "error", "message": "No queries provided."}), 400

            lang_check = []
            suggestions = []

            for query in queries:
                lang_is_valid = False
                suggestions_result = None
                try:
                    # Проверка языка
                    lang = detect(query)
                    if lang == 'en':
                        lang_is_valid = True

                    # Поиск кандидатов через FAISS и реранжирование кандидатов
                    reranked_candidates = pipeline(query, candidate_model)

                    suggestions_result = [(i, id_to_text[i]) for i in reranked_candidates]

                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    return jsonify({"status": "error", "where": "while proccessing queries", "message": str(e), "line_number": exc_tb.tb_lineno}), 500
                
                finally:
                    lang_check.append(lang_is_valid)
                    suggestions.append(suggestions_result)

            return jsonify({"lang_check": lang_check, "suggestions": suggestions})

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            return jsonify({"status": "error", "message": str(e), "where": "while initialization variables", "line_number": exc_tb.tb_lineno}), 500
        
    initialize_models()

    # threading.Thread(target=initialize_models, daemon=True).start()

    # if __name__ == "__main__":
    #     # app.logger.info("Starting ...")
    #     threading.Thread(target=initialize_models, daemon=True).start()
    #     app.run(host="0.0.0.0", port=11000, debug=True)
