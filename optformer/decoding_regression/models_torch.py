# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression models with Pytorch."""

import jaxtyping as jt
import numpy as np
from optformer.decoding_regression import vocabs
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e7


def sample_from_probs(prob_vector: jt.Float[jt.Array, 'P']):
  return np.random.choice(range(len(prob_vector)), p=prob_vector)


# Vectorize the function to apply it over the batch dimension
vectorized_sample = np.vectorize(sample_from_probs, signature='(n)->()')

NEG_INF = float('-inf')

class TransformerLayer(nn.Module):
    def __init__(self, units, num_heads, dropout):
        super().__init__()
        # Use batch_first=True so inputs are [B, L, D]
        self.attn = nn.MultiheadAttention(embed_dim=units,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, units)
        )
        self.norm1 = nn.LayerNorm(units, eps=1e-6)
        self.norm2 = nn.LayerNorm(units, eps=1e-6)

    def forward(self, x):
        # x: [B, L, D]
        B, L, _ = x.size()
        # Create a causal mask: shape [L, L] with -inf in upper triangular part
        causal_mask = torch.triu(torch.full((L, L), NEG_INF, device=x.device), diagonal=1)
        # Self-attention: note that MultiheadAttention expects (query, key, value)
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class AttentionDecoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,         # assumed to be a PyTorch module
        vocab,                      # assumed to have: size, token_length, logit_mask(step), from_int(seq)
        units: int = 128,
        num_layers: int = 1,
        num_heads: int = 1,
        dropout: float = 0.0,
        encoder_dim: int = None,    # dimension of encoder output; must be provided
    ):
        super().__init__()
        if encoder_dim is None:
            raise ValueError("You must specify encoder_dim (the encoder output size).")
        self._encoder = encoder
        self._vocab = vocab
        self._units = units
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout

        # Project encoder output (of size encoder_dim) to units (D)
        self._enc_proj = nn.Linear(encoder_dim, units)

        # Embedding layers for tokens and positions.
        self._token_emb = nn.Embedding(num_embeddings=vocab.size, embedding_dim=units)
        self._pos_emb = nn.Embedding(num_embeddings=vocab.token_length, embedding_dim=units)

        # Build transformer layers.
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(units, num_heads, dropout) for _ in range(num_layers)
        ])

        # Output projection to logits over vocabulary.
        self._output_proj = nn.Linear(units, vocab.size)

    def forward(self, inputs):
        """
        inputs: a tuple (encoder_input, decoded_ids)
          - encoder_input: [B, F] tensor from the encoder.
          - decoded_ids: [B, L_decoded] tensor of token ids.
        """
        encoder_input, decoded_ids = inputs  # unpack inputs
        # Get encoder output and project it: [B, encoder_dim] -> [B, units]
        encoded = self._encoder(encoder_input)
        encoded = self._enc_proj(encoded)
        # Get token embeddings for the decoded tokens: [B, L_decoded] -> [B, L_decoded, units]
        decoded = self._token_emb(decoded_ids)
        # Prepend the encoded context (unsqueezed to [B, 1, units])
        encoded = encoded.unsqueeze(1)
        seq = torch.cat([encoded, decoded], dim=1)  # [B, L, units]

        # Add positional embeddings.
        positions = torch.arange(seq.size(1), device=seq.device)
        pos_embeddings = self._pos_emb(positions)  # [L, units]
        seq = seq + pos_embeddings  # broadcast addition over batch dimension

        # Pass the sequence through transformer layers.
        for layer in self.transformer_layers:
            seq = layer(seq)

        # Project to logits: [B, L, vocab.size]
        logits = self._output_proj(seq)
        return logits

    def decode(self, encoder_input, temperature: float = 1.0, top_k: int = None, top_p: float = None):
        """
        Generates a full sequence using iterative sampling.
        encoder_input: [B, F] tensor.
        Returns a list of decoded outputs (after converting token ids via vocab.from_int).
        """
        B = encoder_input.size(0)
        max_len = self._vocab.token_length
        # Initialize token_ids with -1 (or any invalid index) for shape [B, max_len]
        token_ids = torch.full((B, max_len), -1, dtype=torch.long, device=encoder_input.device)

        for i in range(max_len):
            # For step 0, create an empty decoded_ids tensor.
            if i == 0:
                decoded_ids = torch.empty((B, 0), dtype=torch.long, device=encoder_input.device)
            else:
                decoded_ids = token_ids[:, :i]
            # Forward pass; logits shape: [B, L, vocab.size]
            logits = self.forward((encoder_input, decoded_ids))
            # Get the logits for the most recent time step: [B, vocab.size]
            current_logits = logits[:, -1, :].clone()

            # Apply vocab logit restriction.
            # Assume _vocab.logit_mask(i) returns a boolean array of shape [vocab.size].
            mask = torch.tensor(self._vocab.logit_mask(i), device=current_logits.device, dtype=torch.bool)
            current_logits.masked_fill_(~mask, NEG_INF)

            # Apply top-k filtering if specified.
            if top_k is not None:
                k = min(top_k, current_logits.size(-1))
                kth_vals = current_logits.topk(k, dim=-1).values[:, -1].unsqueeze(-1)
                current_logits = torch.where(current_logits < kth_vals,
                                             torch.full_like(current_logits, NEG_INF),
                                             current_logits)

            # Apply top-p filtering if specified.
            if top_p is not None:
                # Sort logits and compute softmax probabilities.
                sorted_logits, sorted_indices = torch.sort(current_logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits / temperature, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # Mask out tokens with cumulative probability above top_p.
                sorted_mask = cumulative_probs > top_p
                # Shift the mask to keep at least one token.
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                sorted_logits[sorted_mask] = NEG_INF
                # Scatter back to original indexing.
                current_logits = torch.zeros_like(current_logits).scatter_(dim=-1,
                                                                             index=sorted_indices,
                                                                             src=sorted_logits)

            # Compute probabilities and sample the next token.
            probs = F.softmax(current_logits / temperature, dim=-1)
            sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
            token_ids[:, i] = sampled_ids

        # Convert generated token ids to output using vocab.from_int.
        outputs = [self._vocab.from_int(token_ids[b].tolist()) for b in range(B)]
        return outputs

def weighted_sparse_categorical_crossentropy(labels, logits, weights=None):
    """
    Weighted version of sparse categorical cross entropy.

    Args:
        labels (torch.Tensor): Tensor of shape [B, L] with integer class labels.
        logits (torch.Tensor): Tensor of shape [B, L, V] with raw logits.
        weights (torch.Tensor, optional): Weights tensor; expected shape [B, L] or broadcastable
                                          to that shape. If None, no weighting is applied.
    
    Returns:
        torch.Tensor: Loss tensor of shape [B, L] with the weighted loss per token.
    """
    B, L, V = logits.size()
    # Flatten logits and labels to compute loss per token.
    logits_flat = logits.view(-1, V)  # shape [B*L, V]
    labels_flat = labels.view(-1)     # shape [B*L]
    
    # Compute unweighted loss per token.
    ce_flat = F.cross_entropy(logits_flat, labels_flat, reduction='none')
    ce = ce_flat.view(B, L)  # reshape back to [B, L]
    
    if weights is None:
        return ce

    # Ensure weights are floats and normalize along the sequence length.
    weights = weights.float()
    # Multiply by L (the number of tokens) and divide by the sum of weights for each batch.
    normalized_weights = (L * weights) / torch.sum(weights, dim=-1, keepdim=True)
    
    return ce * normalized_weights
