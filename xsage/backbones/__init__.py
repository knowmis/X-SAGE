"""Backbone models and score generators (one runnable module per backbone)."""


def default_device() -> str:
    """"cuda" if available, else "mps" if available, else "cpu"."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
