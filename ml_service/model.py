from typing import Dict, List, Tuple, Callable

import numpy as np
import torch

    
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