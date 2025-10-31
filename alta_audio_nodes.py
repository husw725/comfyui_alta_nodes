import os
import json
from typing import List, Tuple

from pyannote.audio import Pipeline
from pathlib import Path

# ComfyUI imports
import comfy.utils
import comfy.model_management as model_management


class PyannoteSpeakerDiarizationNode:
    """ComfyUI 节点：输入音频文件路径，输出说话人分离结果"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {"default": "vocals_enhanced_silence.wav"}),
                "hf_token": ("STRING", {"default": "your_hf_token_here"}),
                "cache_dir": ("STRING", {"default": "./models/pyannote"}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("speaker_segments",)
    FUNCTION = "run_diarization"
    CATEGORY = "alta/Audio/Analysis"

    def run_diarization(self, audio_path: str, hf_token: str, cache_dir: str) -> Tuple[List]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"[Pyannote] Loading pipeline from cache_dir={cache_dir}")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=hf_token,
            cache_dir=cache_dir
        )

        print(f"[Pyannote] Running diarization on {audio_path}")
        diarization = pipeline(audio_path)

        result = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker
            })

        print(f"[Pyannote] Found {len(result)} segments")
        return (result,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "Alta:SpeakerDiarization": PyannoteSpeakerDiarizationNode
}

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Alta:SpeakerDiarization": "🎙️ Speaker Diarization (Pyannote)"
# }
