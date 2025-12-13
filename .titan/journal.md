## 2024-05-23 - [Optimization/Architecture]
 **Insight:** Concatenating static asset embeddings to the input sequence before projection is memory-inefficient and computationally redundant. It expands the embedding vector to `(Batch, SeqLen, EmbDim)` and performs `SeqLen` redundant matmuls.
 **Rule:** Split the input projection into `Linear(input)` and `Linear(embedding)`. Compute `emb_proj` once per sample `(Batch, 1, D_model)` and broadcast-add to `input_proj` `(Batch, SeqLen, D_model)`. This saves memory and compute.

## 2024-05-23 - [Architecture/Clean Code]
 **Insight:** `MambaRegressor` was wrapping `MambaBlock` in an additional `LayerNorm` and residual connection loop `x = x + mamba(norm(x))`. Since `MambaBlock` already applies `Norm -> Mamba -> Dropout -> Residual`, this created a redundant and potentially harmful "Double Norm / Double Residual" structure.
 **Rule:** Trust the block wrapper. If `MambaBlock` handles residuals/norms, the outer loop should just be `x = block(x)`.
