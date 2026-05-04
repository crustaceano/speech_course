import torch
import torch.nn as nn
import torch.nn.functional as F


class Perplexity(nn.Module):
    EPS = 1e-8
    def __init__(self, n_codecs):
        super().__init__()
        self.n_codecs = n_codecs
    
    def forward(self, indices):
        device = indices.device

        arange = torch.arange(self.n_codecs, device=device)
        indices = indices.flatten()
        encodings = torch.eq(arange.unsqueeze(dim=1), indices.unsqueeze(dim=0))

        probs = torch.mean(encodings.float(), dim=1)
        perplexity = torch.exp(- torch.sum(probs * torch.log(probs + self.EPS)))
        return perplexity


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size

        self.codebook = nn.Embedding(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dim)

        self._init_weight()

    def _init_weight(self):
        init_size = 1 / self.codebook_size
        torch.nn.init.uniform_(self.codebook.weight, a=-init_size, b=init_size)

    def calculate_squared_distances(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        """
        tensor_1: float tensor with shape [sequence_1, embedding] 
        tensor_2: float tensor with shape [sequence_2, embedding]
        output: float tensor with shape [sequence_1, sequence_2]
        """
        tensor_1 = tensor_1[:, None, :]
        tensor_2 = tensor_2[None, :, :]
        distances = ((tensor_1 - tensor_2) ** 2).sum(dim=2)

        return distances

    def encode(self, embeddings: torch.Tensor):
        """
        Encodes the input embeddings, by the indices of closest embeddings from the codebook
        embeddings: Embedded image of size [batch, embedding, height, width]
        output: LongTensor of indices of size [batch, height, width]
        """
        assert embeddings.dim() == 4
        B, E, H, W = embeddings.shape

        embeddings = embeddings.permute(0, 2, 3, 1).reshape(B * H * W, E)
        distances = self.calculate_squared_distances(embeddings, self.codebook.weight)
        indices = torch.argmin(distances, dim=-1, keepdim=False)
        indices = indices.view(B, H, W)
        
        return indices

    def decode(self, indices: torch.Tensor):
        """
        Inserts embeddings from the codebook instead of indices
        Indices: Longtensor of indices from the codebook of size [batch, height, width]
        For each index: 0 <= index < codebook_size
        output: FloatTensor of codec vectors from codebook of size [batch, embedding, height, width]
        """
        decoded = self.codebook(indices)
        decoded = decoded.permute(0, 3, 1, 2)

        return decoded

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantizes embeddings
        """
        indices = self.encode(embeddings)
        quantized = self.decode(indices)

        return quantized


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim, n_codebooks):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks
        
        self.codebooks = [VectorQuantizer(codebook_size, embedding_dim) for _ in range(n_codebooks)]
        self.codebooks = nn.ModuleList(self.codebooks)

    def encode(self, embeddings: torch.Tensor):
        """
        Encodes the input embeddings, by the indices of closest embeddings from the codebook.
        Then iteratively encodes the residuals between the embedding and vectors from the codebook the same way.
        embeddings: Embedded image of size [batch, embedding, height, width]
        output: LongTensor of indices of size [batch, n_codebooks, height, width]
        """
        indices = []
        res = embeddings
        for codebook in self.codebooks:
            level_indices = codebook.encode(res)
            z_level_quatized = codebook.decode(level_indices)
            res = res - z_level_quatized
            indices.append(level_indices)
        
        indices = torch.stack(indices, dim=1)

        return indices

    def decode(self, indices: torch.Tensor):
        """
        Sums the embeddings from the codebooks with dedicated indices
        Indices: Longtensor of indices from the codebook of size [batch, n_codebooks, height, width]
        For each index: 0 <= index < codebook_size
        output: FloatTensor of codec vectors from codebook of size [batch, embedding, height, width]
        """
        quantized = []
        for idx, codebook in enumerate(self.codebooks):
            quantized.append(codebook.decode(indices[:, idx, :]))

        return sum(quantized)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantizes embeddings
        """
        indices = self.encode(embeddings)
        quantized = self.decode(indices)
        return quantized


class VectorQuantizationLoss(nn.Module):
    def __init__(self, commitment_cost=1.):
        super().__init__()
        self.commitment_cost = commitment_cost

    def forward(self, inputs, quantized):
        """
        Calculates the vector quantisation loss
        inputs: vector of embeddings of size [batch, embedding, height, width]
        quantized: the vector of embeddings, processed by VectorQuantisation ot ResidualVectorQuantization
        output: differentiable loss of size [1]
        """

        codeboock_loss = F.mse_loss(inputs.detach(), quantized)
        commitment_loss = F.mse_loss(inputs, quantized.detach())
        loss = codeboock_loss + self.commitment_cost * commitment_loss
        return loss
