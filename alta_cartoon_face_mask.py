import os
import urllib.request
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ============================================================
# Core 模型节点：只负责加载 YOLO 模型
# ============================================================
class YOLOAnimeFaceModel:
    """
    Core 模型节点，负责加载 YOLO 模型
    输出: model 实例
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("yolo_model",)
    FUNCTION = "load_model"
    CATEGORY = "Alta"

    def load_model(self):
        MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
        MODEL_PATH = os.path.join(MODEL_DIR, "yolov8x6_animeface.pt")
        
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_DIR, exist_ok=True)
            model_url = "https://huggingface.co/Fuyucchi/yolov8_animeface/resolve/main/yolov8x6_animeface.pt"
            print(f"[INFO] 模型不存在，正在下载 {MODEL_PATH} ...")
            urllib.request.urlretrieve(model_url, MODEL_PATH)
            print("✅ 模型下载完成！")
        
        model = YOLO(MODEL_PATH)
        print(f"✅ YOLO 模型已加载: {MODEL_PATH}")
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
    "Alta:YOLOAnimeFaceModel": YOLOAnimeFaceModel,
    "Alta:CartoonFaceMask": AltaCartoonFaceMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Alta:YOLOAnimeFaceModel": "Alta YOLO AnimeFace Model",
    "Alta:CartoonFaceMask": "Alta Cartoon Face Mask"
}