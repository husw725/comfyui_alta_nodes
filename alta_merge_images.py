"""
OpenAI API Nodes
API Doc
https://platform.openai.com/docs/guides/image-generation#generate-images
"""

from __future__ import annotations
from openai import OpenAI
import logging
import base64

from comfy_api_nodes.apinode_utils import (
    process_image_response
)
from comfy_api.latest import io as comfy_io
from comfy_api_nodes.apinode_utils import (
    tensor_to_base64_string,
)
from comfy.comfy_types.node_typing import IO, ComfyNodeABC


# def validate_prompts(prompt: str, max_length: int) -> bool:
#     """Verifies that the positive prompt is not empty and that neither promt is too long."""
#     if not prompt:
#         raise ValueError("Positive prompt is empty")
#     if len(prompt) > max_length:
#         raise ValueError(
#             f"Positive prompt is too long: {len(prompt)} characters")
#     return True


# def validate_input_image(image: torch.Tensor) -> None:
#     validate_image_dimensions(image, min_width=320,
#                               min_height=320, max_width=4096, max_height=4096)
#     validate_image_aspect_ratio(
#         image, min_aspect_ratio=1 / 3, max_aspect_ratio=3)


class AltaOpenAIApiError(Exception):
    """Base exception for OpenAI API errors."""

    pass


class OpenAIImages2ImageNode(ComfyNodeABC):
    """
    OpenAI Images Merging Node
    Generate new images using other images as a reference
    """

    FUNCTION = "api_call"
    # API_NODE = True
    OUTPUT_NODE = True
    CATEGORY = "Alta"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = "OpenAI Images Merging Node"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": (
                    IO.STRING, {
                        "tooltip": "OpenAI API Key"
                    }
                ),
                "prompt": (
                    IO.STRING, {
                        "tooltip": "The promotion to describe for the new image.", "multiline": True
                    }
                ),
                "images": (
                    IO.IMAGE,
                ),
            }
        }

    async def api_call(
        self,
        api_key: str,
        prompt: str,
        images: IO.IMAGE,
        **kwargs,
    ) -> str:
        if images is None:
            raise AltaOpenAIApiError("请输入需要合成的参考图片")

        client = OpenAI(api_key=api_key)

        img_params = [{
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{tensor_to_base64_string(i)}",
        } for i in images] + [{"type": "input_text", "text": prompt}]
        params = [{"role": "user", "content": img_params}]
        response = client.responses.create(
            model="gpt-4.1",
            input=params,
            tools=[{"type": "image_generation"}],
        )

        image_generation_calls = [output for output in response.output if output.type == "image_generation_call"]
        image_data = [output.result for output in image_generation_calls]

        if image_data:
            image_base64 = image_data[0]
            return comfy_io.NodeOutput(process_image_response(base64.b64decode(image_base64)))
        else:
            logging.error(response.output.content)
            raise AltaOpenAIApiError()
        

NODE_CLASS_MAPPINGS = {
    "Alta:OpenAIImages2Image": OpenAIImages2ImageNode,
}