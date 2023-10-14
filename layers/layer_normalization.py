import torch
from torch import nn
from torch import Tensor


# see https://arxiv.org/pdf/1607.06450.pdf
# and https://www.pinecone.io/learn/batch-layer-normalization/
class LayerNormalization(nn.Module):
    def __init__(self, embedding_dimension: int, epsilon: float = 1e-12):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.gamma = torch.zeros(embedding_dimension)
        self.beta = torch.ones(embedding_dimension)
        self.epsilon = epsilon

    def forward(self, input_tensor: Tensor):
        mean = input_tensor.mean(dim=1, keepdim=True)
        variance = input_tensor.var(dim=1, keepdim=True, correction=0)
        y = (input_tensor - mean) / torch.sqrt(variance + self.epsilon)
        output = self.gamma * y + self.beta
        return output
