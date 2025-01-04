import os
import sys
import json
import re
import nltk
import numpy as np
import torch
import logging

from typing import Dict, List, Tuple, Callable
from model import KNRM
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

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
        # app.logger.info('Creating vocabulary and model ...')
        fitted_vocab = self.create_vocab_from_file(self.vocab_path)
        torch.manual_seed(self.random_seed)
        knrm = KNRM(embedding_path=self.knrm_embeddings_path,
                    mlp_path=self.mlp_path,
                    freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        # app.logger.info('Created ...')
        return knrm, fitted_vocab

    def predict(self, dataloader):
        all_preds = []
        for batch in (dataloader):
            preds = self.model.predict(batch).squeeze(1)
            preds_np = preds.detach().numpy().tolist()
            all_preds.extend(preds_np)
        return all_preds

def pipeline(query):
    candidates = candidate_model.query(query, size=20)
    candidates = [i[1] for i in candidates]
    ds = RankingDataset(query=query,
                candidates=candidates,
                vocab=ml_service.vocab,
                oov_val=ml_service.vocab['OOV'],
                preproc_func=ml_service.simple_preproc)
    dl = torch.utils.data.DataLoader(
        ds, 
        batch_size=ml_service.dataloader_bs, 
        num_workers=0,
        collate_fn=collate_fn, 
        shuffle=False)
    
    score = ml_service.predict(dl)
    return np.array(candidates)[np.argsort(score)]

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
