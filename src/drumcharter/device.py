"""PyTorch device resolution utilities.

Maps user-facing device strings ("auto", "cpu", "cuda") to concrete
device identifiers, with helpful error messages when CUDA is requested
but unavailable.
"""

from __future__ import annotations

VALID_TORCH_DEVICES = ("auto", "cpu", "cuda")


class DeviceError(RuntimeError):
    """Raised when a requested PyTorch device cannot be used."""


def resolve_torch_device(device: str | None, *, purpose: str) -> str:
    """Resolve a user-facing device choice to a concrete PyTorch device."""
    requested = (device or "auto").lower()
    if requested not in VALID_TORCH_DEVICES:
        expected = ", ".join(VALID_TORCH_DEVICES)
        raise DeviceError(f"Unknown device {device!r}. Expected one of: {expected}")

    try:
        import torch
    except ImportError as exc:
        raise DeviceError(
            f"{purpose} requires PyTorch. Install the 'ai' extra: uv sync --extra ai"
        ) from exc

    cuda_available = bool(torch.cuda.is_available())
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise DeviceError(
            f"CUDA was requested for {purpose}, but CUDA is not available. "
            "Use --device cpu or --device auto."
        )
    return requested
