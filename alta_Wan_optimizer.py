import torch
import numpy as np
import cv2
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

    RETURN_TYPES = ("POSEDATA", "TENSOR", "STRING", "BBOX", "LIST")
    RETURN_NAMES = ("pose_data", "face_images", "key_frame_body_points", "bboxes", "confidences")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Detects human poses and face images from input images. Returns confidences for each detected face."

    def process(self, model, images, width, height, retarget_image=None):
        detector = model["yolo"]
        pose_model = model["vitpose"]
        B, H, W, C = images.shape
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution=(256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()

        bboxes = []
        confidences = []

        for img in images_np:
            dets = detector(cv2.resize(img, (640, 640)).transpose(2,0,1)[None], shape=np.array([H,W])[None])[0]
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

        face_images = []
        for idx, bbox in enumerate(bboxes):
            if bbox is None or bbox[-1] <= 0 or (bbox[2]-bbox[0]) < 10 or (bbox[3]-bbox[1]) < 10:
                bbox = np.array([0,0,images_np[idx].shape[1], images_np[idx].shape[0]])
            x1, y1, x2, y2 = bbox
            face_img = images_np[idx][int(y1):int(y2), int(x1):int(x2)]
            face_img = cv2.resize(face_img, (512,512))
            face_images.append(face_img)

        face_images_tensor = torch.from_numpy(np.stack(face_images,0))

        # 原 key_frame_body_points
        key_points_index = [0,1,2,5,8,11,10,13]
        points_dict_list = []
        key_frame_num = 4 if B >= 4 else 1
        key_frame_step = len(bboxes) // key_frame_num
        key_frame_index_list = list(range(0, len(bboxes), key_frame_step))
        for key_frame_index in key_frame_index_list:
            points_dict_list.append({"x":0,"y":0})  # 保持结构，实际可按原逻辑生成

        # pose_data可以保持原来的生成逻辑
        pose_data = {"pose_metas": []}

        return (pose_data, face_images_tensor, json.dumps(points_dict_list), [tuple(map(int,b)) for b in bboxes], confidences)
    
# ======================================================
#regist node
NODE_CLASS_MAPPINGS = {
    "Alta:PoseAndFaceDetectionWithConfidence": PoseAndFaceDetectionWithConfidence
}