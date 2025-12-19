## 2024-05-22 - [Optimizing Training Loops]
**Learning:** Calling `.item()` inside a training loop forces a CPU-GPU synchronization, which can significantly slow down training, especially with small batches.
**Action:** Accumulate loss components as tensors on the GPU (using `.detach()`) and only convert to scalars at the end of the epoch. Also, use `non_blocking=True` for data transfers.
