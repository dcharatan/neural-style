import torch


def is_input_image(image: torch.Tensor):
    """Images should have shape (batch, channels = 3, rows, columns)."""
    return (
        torch.is_tensor(image)
        and image.dtype == torch.float32
        and image.ndim == 4
        and image.shape[1:] == (3, 256, 256)
    )
