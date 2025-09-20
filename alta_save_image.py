import os
import shutil
import datetime
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# from .imagefunc import *

NODE_NAME = 'SaveImagePlus'

class SaveImagePlus:
    def __init__(self):
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"images": ("IMAGE", ),
                     "filepaths": ("LIST",),  
                     "format": (["png", "jpg"],),
                     "quality": ("INT", {"default": 80, "min": 10, "max": 100, "step": 1}),
                     "preview": ("BOOLEAN", {"default": True}),
                     },
                }

    RETURN_TYPES = ()
    FUNCTION = "save_image_plus"
    OUTPUT_NODE = True
    CATEGORY = 'Alta'

    def save_image_plus(self, images, filepaths, format, quality, preview):
        results = []

        for img_tensor, out_path in zip(images, filepaths):
            # tensor → PIL
            i = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            out_dir = os.path.dirname(out_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            if format == "png":
                img.save(out_path, compress_level=(100 - quality) // 10)
            else:
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save(out_path, quality=quality)

            print(f"SaveImagePlus -> saved {out_path}")

            if preview:
                results.append({
                    "filename": os.path.basename(out_path),
                    "subfolder": os.path.basename(out_dir),
                    "type": self.type
                })

        return {"ui": {"images": results}}

class SaveImageWithName:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "filename": ("STRING", {}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "outputs"}),
                "format": ("STRING", {"default": ""}),  # 默认保持原扩展
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "output_path")
    FUNCTION = "save"
    CATEGORY = "Alta"
    OUTPUT_NODE = True

    def _to_pil(self, image):
        # ✅ 用 np.asarray 兼容 numpy 2.0
        arr = np.asarray(image)

        # 处理 batch 维度 (B,H,W,C)
        if arr.ndim == 4:
            arr = arr[0]  # 取第一张

        # 归一化到 0-255
        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)

        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        if arr.shape[2] == 4:
            return Image.fromarray(arr, mode="RGBA")
        return Image.fromarray(arr, mode="RGB")

    def save(self, image, filename, output_dir="outputs", format=""):
        os.makedirs(output_dir, exist_ok=True)
        name, ext = os.path.splitext(filename)
        if not ext:
            ext = ".png"
        if format:
            ext = f".{format.strip().lower()}"
        out_path = os.path.join(output_dir, f"{name}{ext}")
        pil = self._to_pil(image)
        pil.save(out_path)
        return (image, out_path)


NODE_CLASS_MAPPINGS = {
    
}

NODE_CLASS_MAPPINGS = {
    "Alta:SaveImagePlus": SaveImagePlus,
    "Alta:SaveImage": SaveImageWithName
}
