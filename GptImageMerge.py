import os
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np
from openai import OpenAI


class GptImageMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),  # 第一张，必选
                "prompt": ("STRING", {"multiline": True, "default": "Describe how to merge the images (e.g. 'Put the second image inside the first')."}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge"
    CATEGORY = "Alta/OpenAI"

    def _tensor_to_pil(self, tensor):
        """把 ComfyUI IMAGE tensor 转成 PIL"""
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            arr = (tensor.cpu().numpy().clip(0, 1) * 255).astype("uint8")
            if arr.shape[-1] == 4:
                img = Image.fromarray(arr, "RGBA")
            else:
                img = Image.fromarray(arr, "RGB")
            return img
        raise TypeError(f"Unsupported type: {type(tensor)}")

    def merge(self, image1, prompt, image2=None, image3=None, image4=None, api_key=""):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("⚠️ 请设置 OPENAI_API_KEY 或传入 api_key 参数")

        client = OpenAI(api_key=api_key)

        # 转换 ComfyUI tensor → PIL → BytesIO
        def to_file(img, name):
            if img is None:
                return None
            pil = self._tensor_to_pil(img)
            buf = BytesIO()
            pil.save(buf, format="PNG")
            buf.seek(0)
            return buf

        image_files = []
        for idx, tensor in enumerate([image1, image2, image3, image4], start=1):
            if tensor is not None:
                image_files.append(open(to_file(tensor, f"image{idx}.png").name, "rb") if hasattr(to_file(tensor, f"image{idx}.png"), "name") else to_file(tensor, f"image{idx}.png"))

        # 调用 OpenAI
        result = client.images.edit(
            model="gpt-image-1",
            image=image_files,
            prompt=prompt,
            size="1024x1024"
        )

        # 获取结果
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        out_img = Image.open(BytesIO(image_bytes)).convert("RGB")

        # 转回 ComfyUI IMAGE tensor
        arr = torch.from_numpy(np.array(out_img).astype("float32") / 255.0).unsqueeze(0)

        return (arr,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "Alta:GptImageMerge": GptImageMerge
}

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "GptImageMerge": "Alta:GPT Image Merge (OpenAI)"
# }