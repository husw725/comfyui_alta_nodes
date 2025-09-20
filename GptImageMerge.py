import base64
import io
import requests
from PIL import Image
import torch

class GPTImageMerge:
    """
    使用 OpenAI gpt-image-1 API 将多张图片合成一张
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"forceInput": True}),   # 支持多张输入
                "prompt": ("STRING", {"default": "Merge these images into one."}),
                "api_key": ("STRING", {"default": "sk-xxxx"}),  # OpenAI API key
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge_images"
    CATEGORY = "OpenAI"

    def merge_images(self, images, prompt, api_key):
        # 将 ComfyUI 的 torch tensor 图像转为 PIL
        pil_images = []
        for img in images:
            if isinstance(img, torch.Tensor):
                img = (img.numpy() * 255).astype("uint8")
                img = Image.fromarray(img)
            pil_images.append(img)

        # 转为 base64
        image_data = []
        for pil in pil_images:
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            image_data.append({"name": "image", "buffer": img_b64})

        # 调用 OpenAI gpt-image-1
        url = "https://api.openai.com/v1/images/edits"
        headers = {"Authorization": f"Bearer {api_key}"}
        files = [("image", io.BytesIO(base64.b64decode(img["buffer"]))) for img in image_data]
        data = {"model": "gpt-image-1", "prompt": prompt}

        response = requests.post(url, headers=headers, files=files, data=data)
        result = response.json()
        print("GPTImageMerge -> API response:", result)
        # 解析返回的 base64 图片
        image_b64 = result["data"][0]["b64_json"]
        out_img = Image.open(io.BytesIO(base64.b64decode(image_b64)))

        # 转为 ComfyUI 格式 (torch tensor)
        img_tensor = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(out_img.tobytes()))
             .view(out_img.size[1], out_img.size[0], len(out_img.getbands()))
             .numpy().astype("float32") / 255.0)
        )

        return (img_tensor,)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "Alta:GPTImageMerge": GPTImageMerge
}