import einops
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['CustomInfoNCE', 'info_nce']


class CustomInfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query_pos, key_pos, query_neg, key_neg):
        return info_nce(query_pos, key_pos, query_neg, key_neg,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query_pos, key_pos, query_neg, key_neg, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query_pos.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if key_pos.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    # if key_neg is not None:
    #     if negative_mode == 'unpaired' and key_neg.dim() != 2:
    #         raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
    #     if negative_mode == 'paired' and key_neg.dim() != 3:
    #         raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    # if len(query_pos) != len(key_pos):
    #     raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    # if key_neg is not None:
    #     if negative_mode == 'paired' and len(query_pos) != len(key_neg):
    #         raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    # if query_pos.shape[-1] != key_pos.shape[-1]:
    #     raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    # if key_neg is not None:
    #     if query_pos.shape[-1] != key_neg.shape[-1]:
    #         raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query_pos, key_pos, query_neg, key_neg = normalize(query_pos, key_pos, query_neg, key_neg)
    # if negative_keys is not None:
    # Explicit negative keys

    # Cosine between positive pairs
    positive_logit = torch.sum(query_pos * key_pos, dim=1, keepdim=True)
    # Cosine between negative pairs
    b, n, d = query_neg.shape
    query_neg = einops.rearrange(query_neg, 'b n d -> (b n) d')
    key_neg = einops.rearrange(key_neg, 'b n d -> (b n) d')
    negative_logits = torch.sum(query_neg * key_neg, dim=1, keepdim=True)
    negative_logits = einops.rearrange(negative_logits, '(b n) d -> b n d', b=b, n=n)
    negative_logits = negative_logits.squeeze(-1)
    # if negative_mode == 'unpaired':
    #     # Cosine between all query-negative combinations
    #     negative_logits = query @ transpose(negative_keys)
    #
    # elif negative_mode == 'paired':
    #     query = query.unsqueeze(1)
    #     negative_logits = query @ transpose(negative_keys)
    #     negative_logits = negative_logits.squeeze(1)

    # First index in last dimension are the positive samples
    # negative_logits = negative_logits.unsqueeze(0)
    logits = torch.cat([positive_logit, negative_logits], dim=1)
    labels = torch.zeros(len(logits), dtype=torch.long, device=query_pos.device)
    # else:
    #     # Negative keys are implicitly off-diagonal positive keys.
    #
    #     # Cosine between all combinations
    #     logits = query @ transpose(positive_key)
    #
    #     # Positive keys are the entries on the diagonal
    #     labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
