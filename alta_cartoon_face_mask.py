import os
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# ============================================================
# 模型加载（仅一次）
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "yolov8x6_animeface.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ YOLO 模型未找到: {MODEL_PATH}")

try:
    yolo_model = YOLO(MODEL_PATH)
    print(f"✅ YOLO 模型已加载: {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ 无法加载 YOLO 模型: {e}")

# ============================================================
# 节点定义
# ============================================================
class AltaCartoonFaceMask:
    """
    输入: IMAGE (torch.Tensor, [1,H,W,3], float32, 0~1)
    输出: IMAGE (torch.Tensor, [1,H,W,3], float32, 0~1)
    功能: 使用 YOLO 检测卡通人脸，并通过 GrabCut 生成 mask。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("face_mask",)
    FUNCTION = "create_face_mask"
    CATEGORY = "Alta"

    def create_face_mask(self, image: torch.Tensor):
        # ========== [1] Tensor -> numpy ==========
        image = image.squeeze(0)  # [1,H,W,3] -> [H,W,3]
        img_rgb = (image.numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # ========== [2] 人脸检测 ==========
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

        # ========== [3] 转为 ComfyUI 可识别格式 ==========
        mask_rgb = cv2.cvtColor(mask_total, cv2.COLOR_GRAY2RGB)
        mask_tensor = torch.from_numpy(mask_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        return (mask_tensor,)


# ============================================================
# 节点注册
# ============================================================
NODE_CLASS_MAPPINGS = {
    "Alta:CartoonFaceMask": AltaCartoonFaceMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Alta:CartoonFaceMask": "Alta Cartoon Face Mask"
}