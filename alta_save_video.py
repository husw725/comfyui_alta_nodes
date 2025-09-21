import os
import torch
import numpy as np
from PIL import Image
import imageio
import folder_paths

class MultiInput(str):
    def __new__(cls, string, allowed_types="*"):
        res = super().__new__(cls, string)
        res.allowed_types = allowed_types
        return res
    def __ne__(self, other):
        if self.allowed_types == "*" or other == "*":
            return False
        return other not in self.allowed_types

imageOrLatent = MultiInput("IMAGE", ["IMAGE", "LATENT"])
floatOrInt = MultiInput("FLOAT", ["FLOAT", "INT"])

def tensor_to_bytes(tensor):
    """将 torch.Tensor 或 numpy array 转为 HWC uint8 numpy array"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    if tensor.ndim == 3 and tensor.shape[0] in [1, 3, 4]:
        tensor = np.transpose(tensor, (1,2,0))
    if tensor.dtype in [np.float32, np.float64]:
        tensor = np.clip(tensor*255, 0, 255).astype(np.uint8)
    elif tensor.dtype != np.uint8:
        tensor = tensor.astype(np.uint8)
    return tensor

class VideoCombine:
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats = ["mp4", "mov", "avi", "mkv"]
        return {
            "required": {
                "images": (imageOrLatent,),
                "frame_rate": (floatOrInt, {"default": 8, "min": 1, "step": 1}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename": ("STRING", {"default": ""}),
                "output_path": ("STRING", {"default": ""}),
                "format": (["gif", "webp"] + ffmpeg_formats, {}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    CATEGORY = "Alta"
    FUNCTION = "combine_video"

    def combine_video(
        self,
        images=None,
        latents=None,
        frame_rate=8,
        loop_count=0,
        filename="",
        output_path="",
        format="gif",
        pingpong=False,
        save_output=True,
        **kwargs
    ):
        # 支持 latent 输入
        if latents is not None:
            images = latents
        if images is None:
            return ("",)

        # 支持单张 tensor/numpy
        if isinstance(images, (torch.Tensor, np.ndarray)):
            images = [images]

        # 输出路径处理
        if output_path.strip():
            output_dir = output_path
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()

        # 文件名处理
        base_filename = filename.strip() or "VideoOutput"

        # 转为 PIL images
        pil_images = []
        for i, frame in enumerate(images):
            if isinstance(frame, str):
                frame = Image.open(frame).convert("RGB")
            elif isinstance(frame, torch.Tensor) or isinstance(frame, np.ndarray):
                frame = tensor_to_bytes(frame)
                frame = Image.fromarray(frame)
            else:
                raise TypeError(f"Unsupported image type: {type(frame)}")
            pil_images.append(frame)

        if pingpong:
            pil_images = pil_images + pil_images[::-1]

        ext = format.lower()
        file_path = os.path.join(output_dir, f"{base_filename}.{ext}")

        if ext in ["gif", "webp"]:
            pil_images[0].save(
                file_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=int(1000 / frame_rate),
                loop=loop_count
            )
        else:
            frames_np = [np.array(im.convert("RGB")) for im in pil_images]
            imageio.mimsave(file_path, frames_np, fps=frame_rate, format="FFMPEG", codec="libx264")

        return (file_path,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "Alta:VideoCombine": VideoCombine,
}