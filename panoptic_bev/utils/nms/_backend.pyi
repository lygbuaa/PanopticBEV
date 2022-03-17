import torch

### well, def nms() and torch.ops.load_library() just do the same thing
torch.ops.load_library("_backend.cpython-38-x86_64-linux-gnu.so")
# def nms(bbx: torch.Tensor, scores: torch.Tensor, threshold: double, n_max: int64_t) -> torch.Tensor: ...
