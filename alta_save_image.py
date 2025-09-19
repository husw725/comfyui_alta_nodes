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


NODE_CLASS_MAPPINGS = {
    "Alta:SaveImagePlus": SaveImagePlus
}
