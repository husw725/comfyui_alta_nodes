import torch
import numpy as np
import cv2
from tqdm import tqdm

class PoseAndFaceDetectionWithConfidence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": ("POSEMODEL",),
                "width": ("INT", {"default": 832}),
                "height": ("INT", {"default": 480}),
            }
        }

    RETURN_TYPES = ("POSEDATA", "TENSOR", "LIST", "LIST")
    RETURN_NAMES = ("pose_data", "face_images", "bboxes", "confidences")
    FUNCTION = "process"
    CATEGORY = "Alta/Face"

    def process(self, model, images, width, height):
        detector = model["yolo"]
        pose_model = model["vitpose"]
        B, H, W, C = images.shape
        images_np = images.numpy()

        bboxes_list = []
        conf_list = []
        face_images_list = []

        detector.reinit()
        pose_model.reinit()

        for img in tqdm(images_np):
            detections = detector(
                cv2.resize(img, (640, 640)).transpose(2,0,1)[None],
                shape=np.array([H,W])[None]
            )[0]  # list of dicts or tensor [N,6]

            for det in detections:
                if isinstance(det, dict):
                    if det.get("class_id",0) != 0: continue
                    x1, y1, x2, y2 = det["bbox"]
                    conf = det.get("confidence",1.0)
                else:  # tensor row [x1,y1,x2,y2,conf,class_id]
                    if det[5] != 0: continue
                    x1, y1, x2, y2, conf = det[:5]

                x1, y1, x2, y2 = map(int, [x1,y1,x2,y2])
                face_img = cv2.resize(img[y1:y2, x1:x2], (512,512))
                face_images_list.append(face_img)
                bboxes_list.append([x1,y1,x2,y2])
                conf_list.append(float(conf))

        face_images_tensor = torch.from_numpy(np.stack(face_images_list,0))
        detector.cleanup()
        pose_model.cleanup()

        # pose_data可以用pose_model处理images_np生成，和原节点类似
        pose_data = {"pose_metas": []}

        return (pose_data, face_images_tensor, bboxes_list, conf_list)
    
# ======================================================
#regist node
NODE_CLASS_MAPPINGS = {
    "Alta:PoseAndFaceDetectionWithConfidence": PoseAndFaceDetectionWithConfidence
}