## 2024-05-23 - [PyTorch Dataset NumPy Bottleneck]
**Learning:** Mixing NumPy and PyTorch random operations in a  causes significant overhead due to context switching and CPU-GPU synchronization (or just pure CPU inefficiency of converting tensors to numpy for indexing). Replacing  with  and avoiding  calls yielded a ~40% speedup in .
**Action:** Always prefer pure  operations inside , especially for random number generation and masking logic.
## 2024-05-23 - [PyTorch Dataset NumPy Bottleneck]
**Learning:** Mixing NumPy and PyTorch random operations in a Dataset causes significant overhead due to context switching and CPU-GPU synchronization. Replacing np.random with torch.rand and avoiding .numpy() calls yielded a ~40% speedup in __getitem__.
**Action:** Always prefer pure torch operations inside Dataset.__getitem__, especially for random number generation and masking logic.
