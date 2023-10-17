import torch
from torch import nn
from torch import Tensor


# see https://arxiv.org/pdf/1607.06450.pdf
# and https://www.pinecone.io/learn/batch-layer-normalization/
class LayerNormalization(nn.Module):
    def __init__(self, embedding_dimension: int, epsilon: float = 1e-5):
        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.gamma = torch.zeros(embedding_dimension)
        self.beta = torch.ones(embedding_dimension)
        self.epsilon = epsilon

    def forward(self, x: Tensor):
        '''
        by taking mean/variance, we reduce the dimensionality by 1, so these tensors are [batch_size,1,1].
        Thus, for each batch, we have a mean/variance of all x_i's.
        Next, we subtract input_tensor by mean, so for each batch, we subtract each x_i
        in that batch by the mean for that batch, and divide by the root of variance for that batch.
        Thus, we are left with a tensor [batch_size,length,embedding_dimension].
        The gamma and beta tensors are broadcastable, meaning that, although they are of size[embedding_dimension],
        when we element-wise add/multiply to our tensor y [batch_size,length, embedding_dimension],
        we will be able to automatically expand gamma and beta to the correct size.
        This is also why we can keep epsilon as a single value.

        In short, layer normalization allows us to "normalize" (more accurate term is "standardize")
        the tensor before it is passed to the next layer, this way we reduce internal covariate shift.
        '''

        # input_tensor[batch_size, length, embedding_dimension]
        # output[batch_size, length, embedding_dimension]
        mean = x.mean(dim=(1, 2), keepdim=True)
        variance = x.var(dim=(1, 2), keepdim=True, correction=0)

        y = (x - mean) / torch.sqrt(variance + self.epsilon)
        output = self.gamma * y + self.beta
        return output
