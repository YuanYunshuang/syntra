import numpy as np
import torch


# --- NumPy: prefer the new override_repr hook (NumPy ≥2.1) ---
try:
    po = np.get_printoptions()
    if "override_repr" in po:  # NumPy 2.1+
        np.set_printoptions(override_repr=lambda a: f"np.ndarray(shape={a.shape}, dtype={a.dtype})")
    elif hasattr(np, "set_string_function"):  # legacy (<2.0)
        np.set_string_function(lambda a: f"np.ndarray(shape={a.shape}, dtype={a.dtype})", repr=True)
    else:
        # Fallback: try to patch ndarray.__repr__ (may not work on some builds)
        _old_ndarray_repr = np.ndarray.__repr__
        def _ndarray_repr(self):
            return f"np.ndarray(shape={self.shape}, dtype={self.dtype})"
        np.ndarray.__repr__ = _ndarray_repr
except Exception:
    pass

# --- PyTorch: patch Tensor.__repr__ ---
_old_torch_repr = torch.Tensor.__repr__
def _shape_first_repr(self: torch.Tensor) -> str:
    shape = tuple(self.shape)
    s = f"torch.Tensor(shape={shape}, dtype={self.dtype}, device={self.device}"
    if self.requires_grad: s += ", requires_grad=True"
    s += ")"
    return s
torch.Tensor.__repr__ = _shape_first_repr