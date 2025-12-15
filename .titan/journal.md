## 2025-12-15 - Performance
**Insight:** `gc.collect()` and `torch.cuda.empty_cache()` inside the training loop are massive performance killers. They force CPU-GPU synchronization and halt execution to reclaim memory that the allocator usually manages efficiently.
**Rule:** NEVER call `gc.collect()` or `empty_cache()` inside a hot loop (like `train_epoch`) unless debugging a specific, proven memory leak that crashes the run. Trust the PyTorch allocator.
