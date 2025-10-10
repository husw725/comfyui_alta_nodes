import os
import urllib.request
import cv2
import numpy as np
import torch
import urllib.request
from ultralytics import YOLO

class YOLOFaceModelCore:
    NODE_INFO = "选择动漫或真人 YOLO 模型，加载后输出 yolo_model，可复用到其他节点"

    MODEL_OPTIONS = {
       # 只保留存在的 animeface 模型
    "yolov8x6_animeface": (
        "https://huggingface.co/Fuyucchi/yolov8_animeface/resolve/main/yolov8x6_animeface.pt",
        "~142M",
        "精度极高，推理慢"
    ),
    # "yolov8l_animeface": (
    #     "https://huggingface.co/Fuyucchi/yolov8_animeface/resolve/main/yolov8l_animeface.pt",
    #     "~86M",
    #     "精度高，推理中等"
    # ),
    }

    # 节点内缓存模型，保证同一个模型只加载一次
    _cached_models = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    "STRING",
                    {
                        "default": "yolov8x6_animeface",
                        "choices": list(cls.MODEL_OPTIONS.keys())
                    }
                )
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("yolo_model",)
    FUNCTION = "load_model"
    CATEGORY = "Alta"

    def load_model(self, model_name):
        # 如果缓存中有，直接复用
        if model_name in self._cached_models:
            print(f"[INFO] 使用缓存模型: {model_name}")
            return (self._cached_models[model_name],)

        # 获取模型信息
        url, param, desc = self.MODEL_OPTIONS[model_name]
        print(f"[INFO] 选择模型: {model_name}, 参数量: {param}, 描述: {desc}")

        # 确保模型目录存在
        MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
        os.makedirs(MODEL_DIR, exist_ok=True)
        MODEL_PATH = os.path.join(MODEL_DIR, f"{model_name}.pt")

        # 下载模型（如果不存在）
        if not os.path.exists(MODEL_PATH):
            print(f"[INFO] 模型不存在，正在下载 {MODEL_PATH} ...")
            urllib.request.urlretrieve(url, MODEL_PATH)
            print(f"✅ 模型下载完成: {MODEL_PATH}")

        # 加载模型
        model = YOLO(MODEL_PATH)
        self._cached_models[model_name] = model
        print(f"✅ 模型加载成功: {MODEL_PATH}")
        return (model,)


# ============================================================
# CartoonFaceMask 节点：通过参数接收模型
# ============================================================
class AltaCartoonFaceMask:
    """
    输入:
        - image: torch.Tensor [1,H,W,3], float32 0~1
        - yolo_model: MODEL
    输出:
        - face_mask: torch.Tensor [1,H,W,3], float32 0~1
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "yolo_model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("face_mask",)
    FUNCTION = "create_face_mask"
    CATEGORY = "Alta"

    def create_face_mask(self, image: torch.Tensor, yolo_model):
        # Tensor -> numpy
        img_rgb = (image.squeeze(0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # YOLO 检测
        results = yolo_model(img_bgr)[0]
        mask_total = np.zeros((h, w), np.uint8)

        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            rect = (x1, y1, x2 - x1, y2 - y1)

            mask = np.zeros((h, w), np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask_bin = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
            mask_total = cv2.bitwise_or(mask_total, mask_bin)

        # 转回 tensor
        mask_rgb = cv2.cvtColor(mask_total, cv2.COLOR_GRAY2RGB)
        mask_tensor = torch.from_numpy(mask_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        return (mask_tensor,)
    
# ============================================================
# 节点注册
# ============================================================
NODE_CLASS_MAPPINGS = {
    "Alta:YOLOFaceModelCore": YOLOFaceModelCore,
    "Alta:GetFaceMask": AltaCartoonFaceMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Alta:YOLOFaceModelCore": "Alta YOLO Face Model",
    "Alta:GetFaceMask": "Alta Cartoon Face Mask"
}