"""
Alta API Nodes
API Doc
https://www.volcengine.com/docs/85621/1544716
"""

from __future__ import annotations
import requests
import torch
import logging
from comfy_api.latest import io as comfy_io
from comfy_api_nodes.apinode_utils import (
    tensor_to_base64_string,
    download_url_to_video_output,
)
from comfy_api_nodes.util.validation_utils import (
    validate_image_dimensions,
    validate_image_aspect_ratio
)
from comfy.comfy_types.node_typing import IO, ComfyNodeABC


MAX_PROMPT_LENGTH_I2V = 800


def validate_prompts(prompt: str, max_length: int) -> bool:
    """Verifies that the positive prompt is not empty and that neither promt is too long."""
    if not prompt:
        raise ValueError("Positive prompt is empty")
    if len(prompt) > max_length:
        raise ValueError(
            f"Positive prompt is too long: {len(prompt)} characters")
    return True


def validate_input_image(image: torch.Tensor) -> None:
    validate_image_dimensions(image, min_width=320,
                              min_height=320, max_width=4096, max_height=4096)
    validate_image_aspect_ratio(
        image, min_aspect_ratio=1 / 3, max_aspect_ratio=3)


class AltaApiError(Exception):
    """Base exception for Alta API errors."""
    pass


