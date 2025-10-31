import os
import tempfile
from typing import List, Tuple
from pathlib import Path
from pyannote.audio import Pipeline

import numpy as np
import soundfile as sf


class PyannoteSpeakerDiarizationNode:
    """ComfyUI 节点：输入音频文件或上游 audio 对象，输出说话人分离结果"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hf_token": ("STRING", {"default": "your_hf_token_here"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_path": ("STRING", {"default": ""}),
                "cache_dir": ("STRING", {"default": "./models/pyannote"}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("speaker_segments",)
    FUNCTION = "run_diarization"
    CATEGORY = "alta/Audio/Analysis"

    def run_diarization(self, hf_token: str, audio=None, audio_path: str = "", cache_dir: str = "./models/pyannote") -> Tuple[List]:
        # --------------------------
        # 1️⃣ 获取音频文件路径
        # --------------------------
        if audio is not None:
            # 如果是从上一个节点传来的 AUDIO 对象（tuple: (waveform, sample_rate)）
            waveform, sample_rate = audio
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp_file.name, waveform.T if waveform.ndim > 1 else waveform, sample_rate)
            tmp_file.flush()
            audio_file = tmp_file.name
            print(f"[Pyannote] Using temp audio file: {audio_file}")
        elif audio_path and os.path.exists(audio_path):
            audio_file = audio_path
            print(f"[Pyannote] Using audio file: {audio_path}")
        else:
            raise ValueError("Please provide either 'audio' input or a valid 'audio_path'.")

        # --------------------------
        # 2️⃣ 加载 pyannote pipeline
        # --------------------------
        print(f"[Pyannote] Loading model to cache_dir={cache_dir}")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=hf_token,
            cache_dir=cache_dir
        )

        # --------------------------
        # 3️⃣ 执行分离
        # --------------------------
        print(f"[Pyannote] Running diarization on {audio_file}")
        diarization = pipeline(audio_file)

        result = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker
            })

        print(f"[Pyannote] Found {len(result)} segments")

        # --------------------------
        # 4️⃣ 清理临时文件
        # --------------------------
        if audio is not None:
            os.remove(audio_file)

        return (result,)
# 注册节点
NODE_CLASS_MAPPINGS = {
    "Alta:SpeakerDiarization": PyannoteSpeakerDiarizationNode
}

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Alta:SpeakerDiarization": "🎙️ Speaker Diarization (Pyannote)"
# }
