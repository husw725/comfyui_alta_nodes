import os
import folder_paths
import subprocess
import shutil
import numpy as np
from PIL import Image

def get_video_formats():
    ffmpeg_formats = {
        "video/mp4": [["libx264", "libx265", "mpeg4"], "H.264/H.265/MPEG4"],
        "video/webm": [["libvpx-vp9", "libvpx"], "VP8/VP9"],
        "video/avi": [["mpeg4"], "MPEG4 AVI"],
    }
    format_widgets = {}
    for mime, (codecs, label) in ffmpeg_formats.items():
        format_widgets[mime] = [[codec for codec in codecs], {"default": codecs[0]}]
    return ffmpeg_formats, format_widgets

class VideoCombine:
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats, format_widgets = get_video_formats()
        format_widgets["image/webp"] = [['lossless', "BOOLEAN", {'default': True}]]

        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "INT",
                    {"default": 8, "min": 1, "max": 60, "step": 1},
                ),
                "format": (list(ffmpeg_formats.keys()) + ["image/gif", "image/webp"],),
                "audio": ("AUDIO", {"default": None}),
                "filename": (
                    "STRING",
                    {"default": "output", "multiline": False},
                ),  # 新增参数：保存文件名
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "combine"
    CATEGORY = "Alta/video"

    def combine(self, images, frame_rate, format, audio=None, filename="output"):
        # 确保输出文件夹
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        # 拼接文件路径（避免重复扩展名）
        ext = format.split("/")[-1]
        output_file = os.path.join(output_dir, f"{filename}.{ext}")

        # 转 numpy -> 图片帧
        frames = []
        for img in images:
            img = Image.fromarray(np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8))
            frames.append(img)

        # 保存为不同格式
        if format == "image/gif":
            frames[0].save(output_file, save_all=True, append_images=frames[1:], duration=int(1000/frame_rate), loop=0)
        elif format == "image/webp":
            frames[0].save(output_file, save_all=True, append_images=frames[1:], duration=int(1000/frame_rate), loop=0, lossless=True)
        else:
            # 视频格式 -> 先存临时帧，再 ffmpeg 合并
            tmp_dir = os.path.join(output_dir, "_tmp_frames")
            os.makedirs(tmp_dir, exist_ok=True)

            for i, f in enumerate(frames):
                f.save(os.path.join(tmp_dir, f"frame_{i:05d}.png"))

            cmd = [
                "ffmpeg",
                "-y",
                "-framerate", str(frame_rate),
                "-i", os.path.join(tmp_dir, "frame_%05d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
            ]
            if audio is not None:
                audio_path = folder_paths.get_annotated_filepath(audio)
                cmd += ["-i", audio_path, "-c:a", "aac", "-shortest"]

            cmd += [output_file]
            subprocess.run(cmd, check=True)

            shutil.rmtree(tmp_dir)

        return (output_file,)

NODE_CLASS_MAPPINGS = {
    "Alta:VideoCombine": VideoCombine,
}
