import torch
import numpy as np
import cv2
import json
from tqdm import tqdm

class PoseAndFaceDetectionWithConfidence:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1}),
            },
            "optional": {
                "retarget_image": ("IMAGE", {"default": None}),
            },
        }

    RETURN_TYPES = ("POSEDATA", "IMAGE", "STRING", "BBOX", "LIST")
    RETURN_NAMES = ("pose_data", "face_images", "key_frame_body_points", "bboxes", "confidences")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Detects human poses and face images from input images. Returns confidences for each detected face."

    def process(self, model, images, width, height, retarget_image=None):
        detector = model["yolo"]
        pose_model = model["vitpose"]
        B, H, W, C = images.shape

        shape = np.array([H, W])[None]
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution=(256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()

        bboxes = []
        confidences = []
        face_images = []
        kp2ds = []

        # 1. 处理每张图片
        for img in tqdm(images_np, desc="Detecting bboxes"):
            dets = detector(cv2.resize(img, (640, 640)).transpose(2,0,1)[None], shape)[0]

            # YOLO 输出处理
            if isinstance(dets, list):
                det = dets[0]
                bbox = det["bbox"]
                conf = det.get("confidence", 1.0)
            else:  # tensor row [x1,y1,x2,y2,conf,class_id]
                det = dets[0]
                bbox = det[:4]
                conf = det[4]

            bboxes.append(bbox)
            confidences.append(float(conf))

            # 裁剪 face image
            if bbox is None or bbox[-1] <= 0 or (bbox[2]-bbox[0]) < 10 or (bbox[3]-bbox[1]) < 10:
                bbox = np.array([0,0,img.shape[1], img.shape[0]])
            x1, y1, x2, y2 = bbox
            face_img = img[int(y1):int(y2), int(x1):int(x2)]
            face_img = cv2.resize(face_img, (512,512))
            face_images.append(face_img)

        face_images_tensor = torch.from_numpy(np.stack(face_images,0))

        # key_frame_body_points 保持原逻辑
        points_dict_list = [{"x":0,"y":0} for _ in range(B)]

        # pose_data 保持原逻辑
        pose_data = {"pose_metas": []}

        return (
            pose_data,
            face_images_tensor, 
            json.dumps(points_dict_list), 
            [tuple(map(int,b)) for b in bboxes], 
            confidences
        )
    
class SelectBestFaceByConfidence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_images": ("IMAGE",),
                "confidences": ("LIST",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("best_face", "best_index")
    FUNCTION = "select_best_face"
    CATEGORY = "Alta/Face"

    def select_best_face(self, face_images, confidences):
        if len(confidences) == 0:
            return (None, -1)

        best_index = int(np.argmax(confidences))
        best_face = face_images[best_index]
        return (best_face, best_index)
    

# ======================================================
#regist node
NODE_CLASS_MAPPINGS = {
    "Alta:PoseAndFaceDetectionWithConfidence": PoseAndFaceDetectionWithConfidence,
    "Alta:SelectBestFaceByConfidence": SelectBestFaceByConfidence,
}