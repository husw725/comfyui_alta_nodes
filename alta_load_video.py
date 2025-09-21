import os
import hashlib

# 简单类型替代，保持 ComfyUI 节点兼容
imageOrLatent = "IMAGE"
video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
BIGMAX = 0xFFFFFFFF
DIMMAX = 16384

def simple_hash(path: str):
    """返回路径的 hash 值"""
    return hashlib.md5(path.encode("utf-8")).hexdigest()

def validate_path(path: str, allow_none: bool = False) -> bool:
    if path is None:
        return allow_none
    return os.path.isfile(path)

def load_video_stub(video, **kwargs):
    """
    假实现：返回 IMAGE_DATA, frame_count, audio, video_info
    实际可以替换成你的 load_video
    """
    frame_count = 100
    audio = None
    video_info = {
        "path": video,
        "fps": kwargs.get("force_rate", 0),
        "size": (kwargs.get("custom_width", 0), kwargs.get("custom_height", 0))
    }
    return "IMAGE_DATA", frame_count, audio, video_info

def list_video_files(folder: str):
    """返回文件夹里所有视频路径"""
    if not os.path.isdir(folder):
        raise ValueError(f"{folder} is not a folder")
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in video_extensions
    ]
    files.sort()
    return files

# ===================== 单视频节点 =====================
class LoadVideoPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("STRING", {"placeholder": "X://insert/path/here.mp4", "vhs_path_extensions": video_extensions}),
                "force_rate": ("FLOAT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": DIMMAX}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": DIMMAX}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
            },
            "optional": {},
            "hidden": {
                "force_size": "STRING",
                "unique_id": "UNIQUE_ID"
            },
        }

    CATEGORY = "Alta"
    RETURN_TYPES = (imageOrLatent, "INT", "AUDIO", "VHS_VIDEOINFO", "STRING")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info", "filename")
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        video_path = kwargs["video"]
        if not validate_path(video_path):
            raise Exception(f"video is not a valid path: {video_path}")

        # 调用 stub 或实际 load_video
        image, frame_count, audio, video_info = load_video_stub(
            video=video_path,
            force_rate=kwargs.get("force_rate", 0),
            custom_width=kwargs.get("custom_width", 0),
            custom_height=kwargs.get("custom_height", 0)
        )

        # 单视频直接返回文件名字符串
        filename = os.path.basename(video_path)
        return image, frame_count, audio, video_info, filename

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        return simple_hash(video)

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        return validate_path(video, allow_none=True)

# ===================== 批量视频节点 =====================
class LoadVideosFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"placeholder": "X://insert/folder/path"}),
            },
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    CATEGORY = "Alta"
    RETURN_TYPES = ("LIST", "INT")
    RETURN_NAMES = ("video_paths", "video_count")
    FUNCTION = "load_videos"

    def load_videos(self, folder, **kwargs):
        video_paths = list_video_files(folder)
        video_count = len(video_paths)
        return video_paths, video_count

    @classmethod
    def IS_CHANGED(cls, folder, **kwargs):
        return simple_hash(folder)

    @classmethod
    def VALIDATE_INPUTS(cls, folder):
        return os.path.isdir(folder)

# ===================== 节点映射 =====================
NODE_CLASS_MAPPINGS = {
    "Alta:LoadVideoPath": LoadVideoPath,
    "Alta:LoadVideosFromFolder": LoadVideosFromFolder,
}